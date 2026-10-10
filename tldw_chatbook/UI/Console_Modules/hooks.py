"""DOM-free hook review and one captured Console Send continuation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any

from textual.worker import NoActiveWorker, get_current_worker

from tldw_chatbook.Chat.console_hook_review import HookReviewResult as HookReviewResult

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


def in_worker_task() -> bool:
    """Return whether the running task IS a Textual worker's own task.

    ``get_current_worker()`` alone is not enough: it reads a contextvar, and
    Textual starts a screen's message pump with ``create_task``, which copies
    the caller's contextvars. A Console pushed by tab navigation (which runs
    in the app's ``screen-navigation`` worker) therefore answers "in a
    worker" in every handler on its pump, and a Send there awaited the whole
    review on that pump (TASK-33621.28 review). The worker's task is
    Textual 8's private ``Worker._task``; if a Textual upgrade drops it, this
    answers False -- the safe side (hand off), which the spoken-send tests
    in ``Tests/UI/test_console_hook_review_send_freeze.py`` would report.

    ``_task`` is None only during a worker's FIRST step: Textual's
    ``App.run_async`` installs ``asyncio.eager_task_factory`` (``run_test``
    does not), so that step runs inside ``create_task``, before the result
    is assigned. A running worker with no task yet is therefore the current
    one. (A pump started in that very step also inherits it, but only its
    own first step could see it so: the worker is assigned its task before
    that pump's task runs again.)

    Returns:
        True when the calling task is the current worker's own task, or when
        that worker has no task yet (its first step under the eager factory),
        so awaiting a review there blocks no message pump. The one pump that
        also sees True is one started inside that same first step, and only
        during its own first step, as above. False on every other pump --
        including one that merely inherited a worker's contextvar -- and when
        there is no worker or Textual no longer exposes ``Worker._task``.
    """
    try:
        worker = get_current_worker()
    except NoActiveWorker:
        return False
    if not hasattr(worker, "_task"):
        return False
    task = worker._task
    if task is None:
        return worker.is_running
    return task is asyncio.current_task()


@dataclass(slots=True)
class _HookRefreshFlight:
    generation: int
    sequence: int
    accessor: Callable[[], HookPermissions]
    publisher: Callable[[HookReviewSnapshot], None]
    owner: HookPermissions
    reader: Callable[[], HookReviewSnapshot]
    producer: asyncio.Task[HookReviewSnapshot]
    published: bool = False
    publication_error: BaseException | None = None


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
        session_label: Callable[[str], str] | None = None,
        defer_preparation: Callable[[], bool] | None = None,
        on_send_settled: Callable[[], None] | None = None,
    ) -> None:
        """Wire the controller to its owners.

        Args:
            start_worker: Runs a Send's review-then-dispatch continuation in a
                Textual worker. When given, ``dispatch`` called outside a
                worker hands its review to it instead of awaiting the review
                itself (TASK-33621.28). ``None`` keeps every review inline,
                for callers that own their own task.
            session_label: Names a chat for the "already in progress"
                refusal when the busy Send is another tab's (TASK-33620.15.2).
            defer_preparation: Identifies the actual input message pumps. Their
                complete fresh preparation runs in the worker, including Sends
                whose hooks are already ready. Other callers retain their
                settled inline result.
            on_send_settled: Schedules one deferred indicator refresh after a
                Send releases its complete preparation and dispatch custody.
        """
        self._permissions = hook_permissions_accessor
        self._review = request_review
        self._session = current_session
        self._stash = current_stash
        self._on_state = on_state
        self._notify = notify
        self._start_worker = start_worker
        self._session_label = session_label
        self._generation = 0
        self._busy = False
        #: The chat whose Send holds ``_busy``; ``None`` for a plain review.
        self._busy_session: str | None = None
        self._defer_preparation = defer_preparation
        self._on_send_settled = on_send_settled
        self._send_in_progress = False
        self._pending_send_identity: tuple[str, int] | None = None
        self._refresh_deferred = False
        self._refresh_sequence = 0
        self._refresh_flight: _HookRefreshFlight | None = None
        self.review_open = False

    @property
    def pending_send_identity(self) -> tuple[str, int] | None:
        """Return the pinned Send identity only while its generation is current."""
        identity = self._pending_send_identity
        return identity if identity and identity[1] == self._generation else None

    async def _request_review(self, snapshot, waiting):
        self.review_open = True
        try:
            return await self._review(snapshot, waiting, self.cancel_pending)
        finally:
            self.review_open = False

    def cancel_pending(self) -> None:
        """Invalidate a continuation synchronously before navigation/dismissal."""
        self._generation += 1

    @staticmethod
    def _same_reader(current, captured) -> bool:
        if isinstance(current, MethodType) and isinstance(captured, MethodType):
            return (
                current.__self__ is captured.__self__
                and current.__func__ is captured.__func__
            )
        return current is captured

    def _refresh_current(self, flight: _HookRefreshFlight) -> bool:
        if (
            self._generation != flight.generation
            or self._refresh_sequence != flight.sequence
            or self._permissions is not flight.accessor
            or self._on_state is not flight.publisher
        ):
            return False
        try:
            owner = flight.accessor()
        except RuntimeError:  # The runtime may have been disposed meanwhile.
            return False
        return (
            owner is flight.owner
            and self._same_reader(owner.visit_snapshot, flight.reader)
            and self._generation == flight.generation
            and self._permissions is flight.accessor
            and self._on_state is flight.publisher
        )

    async def refresh(self) -> None:
        """Join one in-flight indicator read and retain its physical lifetime."""
        while True:
            if self._send_in_progress:
                self._refresh_deferred = True
                return
            accessor, publisher = self._permissions, self._on_state
            generation = self._generation
            try:
                owner = accessor()
            except RuntimeError:  # Disposal can precede a queued UI refresh.
                return
            reader = owner.visit_snapshot
            flight = self._refresh_flight
            if flight is not None and flight.producer.done():
                self._refresh_flight = None
                flight = None
            if flight is not None and not (
                flight.generation == generation
                and flight.accessor is accessor
                and flight.publisher is publisher
                and flight.owner is owner
                and self._same_reader(reader, flight.reader)
            ):
                # A different owner waits for the actual old reader, then
                # recaptures every source rather than retaining a stale waiter.
                try:
                    await self._owned_result(flight.producer, discard_source_error=True)
                finally:
                    if self._refresh_flight is flight and flight.producer.done():
                        self._refresh_flight = None
                continue
            if flight is None:
                self._refresh_sequence += 1
                flight = _HookRefreshFlight(
                    generation,
                    self._refresh_sequence,
                    accessor,
                    publisher,
                    owner,
                    reader,
                    asyncio.create_task(asyncio.to_thread(reader)),
                )
                self._refresh_flight = flight
            try:
                snapshot = await self._owned_result(flight.producer)
                if flight.publication_error is not None:
                    raise flight.publication_error
                if not flight.published and self._refresh_current(flight):
                    flight.published = True
                    try:
                        flight.publisher(snapshot)
                    except BaseException as error:
                        flight.publication_error = error
                        raise
                return
            finally:
                if self._refresh_flight is flight and flight.producer.done():
                    self._refresh_flight = None

    def _finish_send(self, session_id: str, generation: int) -> None:
        if self._pending_send_identity != (session_id, generation):
            return
        self._pending_send_identity = None
        self._send_in_progress = False
        self._busy = False
        if self._refresh_deferred:
            self._refresh_deferred = False
            if self._on_send_settled is not None:
                try:
                    self._on_send_settled()
                except (Exception, asyncio.CancelledError):
                    # Indicator scheduling cannot change an actual Send outcome.
                    self._refresh_deferred = True

    async def review_current(self) -> None:
        if self._busy:
            return
        self._busy, self._busy_session = True, None
        try:
            snapshot = await self._owned_snapshot(self._permissions().snapshot)
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

        Called anywhere but a worker's own task -- the task Enter, Send and
        the Workbench's send run in (TASK-33620.15), or a handler on a pump
        -- a Send that needs review returns ``AWAITING_REVIEW`` at once and
        its review-then-dispatch continuation runs in a worker. The app pump
        must stay free: it delivers every key and click the review needs, so
        awaiting the review there froze the whole app, Ctrl+Q included
        (TASK-33621.28). A worker
        caller (spoken "send") awaits the whole continuation and gets its
        settled outcome. ``in_worker_task`` decides which, by task identity.
        """
        if self._busy:
            return self._refused(session_id, self._busy_copy(session_id))
        self._busy, self._busy_session = True, session_id
        self._send_in_progress = True
        self._generation += 1
        generation = self._generation
        self._pending_send_identity = (session_id, generation)
        owns_busy = True
        try:
            if (
                self._start_worker is not None
                and self._defer_preparation is not None
                and self._defer_preparation()
            ):
                # Transfer before the FIRST native await: an awaiting pump
                # cannot deliver further input even if its reader is off-loop.
                owner = self._permissions()
                reader = owner.snapshot
                continuation = self._prepare_in_worker(
                    owner, reader, generation, session_id, stash, dispatch
                )
                try:
                    self._start_worker(continuation)
                except BaseException:
                    continuation.close()
                    raise
                owns_busy = False
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.AWAITING_REVIEW, session_id
                )
            snapshot = await self._owned_snapshot(self._permissions().snapshot)
            self._on_state(snapshot)
            if (
                not snapshot.ready
                and self._start_worker is not None
                and not in_worker_task()
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
                self._finish_send(session_id, generation)

    def _captured_send_current(
        self, owner, reader, generation, session_id, stash
    ) -> bool:
        if (
            generation != self._generation
            or self._session() != session_id
        ):
            return False
        try:
            if self._permissions() is not owner:
                return False
        except RuntimeError:
            return False
        current = owner.snapshot
        if isinstance(reader, MethodType) and isinstance(current, MethodType):
            return (
                current.__self__ is reader.__self__
                and current.__func__ is reader.__func__
            )
        return current is reader

    @staticmethod
    async def _owned_snapshot(reader):
        """Keep an accepted native reader alive until physical retirement."""
        producer = asyncio.create_task(asyncio.to_thread(reader))
        return await ConsoleHooksController._owned_result(producer)

    @staticmethod
    async def _owned_result(producer, *, discard_source_error=False):
        """Drain a shared native producer through repeated caller cancellation."""
        cancelled = None
        while not producer.done():
            try:
                await asyncio.shield(producer)
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if not (
                    discard_source_error
                    and producer.cancelled()
                    and task is not None
                    and task.cancelling() == 0
                ):
                    cancelled = error
            except BaseException:
                if cancelled is None and not discard_source_error:
                    raise
        if cancelled is not None:
            # Consume any producer failure; caller cancellation takes precedence.
            try:
                producer.result()
            except BaseException:
                pass
            raise cancelled
        try:
            return producer.result()
        except (Exception, asyncio.CancelledError):
            if discard_source_error:
                return None
            raise

    async def _prepare_in_worker(
        self, owner, reader, generation, session_id, stash, dispatch
    ) -> ConsolePromptDispatchResult:
        try:
            from tldw_chatbook.Chat.console_send_diagnostics import (
                send_diagnostic_scope,
            )

            async with send_diagnostic_scope("hook_review_continuation") as diagnostic:
                snapshot = await self._owned_snapshot(reader)
                if not self._captured_send_current(
                    owner, reader, generation, session_id, stash
                ):
                    result = self._refused(
                        session_id, "Draft, chat or hooks changed; Send again."
                    )
                else:
                    self._on_state(snapshot)
                    result = await self._continue(
                        snapshot,
                        generation,
                        session_id,
                        stash,
                        dispatch,
                        permissions_owner=owner,
                        snapshot_reader=reader,
                    )
                diagnostic.outcome = result.status.value
                return result
        finally:
            self._finish_send(session_id, generation)

    async def _continue_in_worker(
        self,
        snapshot: HookReviewSnapshot,
        generation: int,
        session_id: str,
        stash: ConsoleDraftStash | None,
        dispatch: Callable[[], Awaitable[ConsolePromptDispatchResult]],
    ) -> ConsolePromptDispatchResult:
        try:
            # Inside the try: nothing before it may skip releasing the Send.
            from tldw_chatbook.Chat.console_send_diagnostics import (
                send_diagnostic_scope,
            )

            # The worker inherits the Send's diagnostic attempt, whose UI
            # scopes closed as awaiting_review; record how the Send ended.
            async with send_diagnostic_scope("hook_review_continuation") as diagnostic:
                result = await self._continue(
                    snapshot, generation, session_id, stash, dispatch
                )
                diagnostic.outcome = result.status.value
                return result
        finally:
            self._finish_send(session_id, generation)

    async def _continue(
        self,
        snapshot: HookReviewSnapshot,
        generation: int,
        session_id: str,
        stash: ConsoleDraftStash | None,
        dispatch: Callable[[], Awaitable[ConsolePromptDispatchResult]],
        *,
        permissions_owner: HookPermissions | None = None,
        snapshot_reader: Callable[[], HookReviewSnapshot] | None = None,
    ) -> ConsolePromptDispatchResult:
        reviewed = not snapshot.ready
        if reviewed:
            result = await self._request_review(snapshot, True)
            if result.kind != "ready":
                return self._refused(session_id, "Send cancelled; draft kept.")
            if permissions_owner is None:
                snapshot = await self._owned_snapshot(self._permissions().snapshot)
            else:
                if not self._captured_send_current(
                    permissions_owner, snapshot_reader, generation, session_id, stash
                ):
                    return self._refused(
                        session_id, "Draft, chat or hooks changed; Send again."
                    )
                snapshot = await self._owned_snapshot(snapshot_reader)
                if not self._captured_send_current(
                    permissions_owner, snapshot_reader, generation, session_id, stash
                ):
                    return self._refused(
                        session_id, "Draft, chat or hooks changed; Send again."
                    )
            self._on_state(snapshot)
        # The chat is always re-checked: the dispatcher reads the visible
        # chat's send gate next. The draft only after a review: keys flow
        # during the snapshot read, and text typed after the capture belongs
        # to the next draft (TASK-340), not a reason to refuse this send.
        if (
            not snapshot.ready
            or generation != self._generation
            or self._session() != session_id
            or (reviewed and not same_captured_draft(self._stash(), stash))
            or (
                permissions_owner is not None
                and not self._captured_send_current(
                    permissions_owner, snapshot_reader, generation, session_id, stash
                )
            )
        ):
            return self._refused(
                session_id, "Draft, chat or hooks changed; Send again."
            )
        # The captured continuation is consumed before the normal dispatcher awaits.
        self._generation += 1
        return await dispatch()

    def _busy_copy(self, session_id: str) -> str:
        """The "already in progress" refusal, naming another tab's busy Send."""
        busy = self._busy_session
        if busy is None or busy == session_id or self._session_label is None:
            return "Hook review or Send is already in progress."
        label = self._session_label(busy)
        return f"Hook review or Send is already in progress in “{label}”."

    def _refused(self, session_id: str, detail: str) -> ConsolePromptDispatchResult:
        self._notify(detail, "warning")
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.REFUSED, session_id, detail
        )
