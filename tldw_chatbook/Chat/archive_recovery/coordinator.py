"""One-shot app confirmation and retained recovery operation custody (ADR-166)."""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from tldw_chatbook.Utils.input_validation import validate_workspace_name

from .contracts import (
    ArchiveRecoveryError,
    ArchiveTarget,
    RecoveryCallKey,
    RecoveryConfirmation,
    RecoveryOperationSnapshot,
    RecoveryOutcome,
    RecoveryPhase,
    RestoreScope,
)
from .operations import (
    allowed_restore_scopes,
    execute_restore,
    read_recovery_target,
    recovery_outcome,
)
from .result_store import ArchiveResultStore

CONFIRMATION_TTL_SECONDS = 300
MAX_SESSION_RECEIPTS = 20


@dataclass
class _Operation:
    operation_id: str
    session_id: str
    target: ArchiveTarget | None
    call_key: RecoveryCallKey | None
    ready: asyncio.Future
    result: asyncio.Future
    admitted: bool = False
    phase: RecoveryPhase = "awaiting_confirmation"
    confirmation: RecoveryConfirmation | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task | None = None
    timer: asyncio.TimerHandle | None = None
    request_ids: set[str] = field(default_factory=set)


class ArchiveRecoveryCoordinator:
    """Own storage until settlement, independent of waiters and mounted views.

    All methods are used on the owning asyncio loop. Native identity registration,
    UI callbacks and decisions are app-only seams, never tool parameters.
    """

    def __init__(
        self,
        app: Any,
        results: ArchiveResultStore,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._app = app
        self._results = results
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._active: dict[str, _Operation] = {}
        self._operations: dict[str, _Operation] = {}
        self._requests: dict[str, str] = {}
        self._receipts: dict[str, OrderedDict[str, None]] = {}
        self._runs: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
        self._closed_runs: set[tuple[str, str]] = set()
        self._closed_sessions: set[str] = set()
        self._sessions: set[str] = set()
        self._closing = False
        self._tasks: dict[asyncio.Task, _Operation] = {}
        self._session_closures: dict[str, asyncio.Task] = {}
        self._shutdown_task: asyncio.Task | None = None
        self._present: Callable[[RecoveryConfirmation], bool] | None = None
        self._is_answerable: Callable[[str], bool] | None = None
        self._on_settled: Callable[[RecoveryOperationSnapshot], None] | None = None

    def set_presenter(
        self,
        present: Callable[[RecoveryConfirmation], bool] | None,
        is_answerable: Callable[[str], bool] | None,
        *,
        on_settled: Callable[[RecoveryOperationSnapshot], None] | None = None,
    ) -> None:
        """Bind app-owned display/answerability and optional completion delivery.

        ``present`` returns True only when exact controls were mounted/retained by
        an answerable UI. ``is_answerable`` checks the source session, not merely
        app existence. Background remount reads ``snapshot``/``present_pending``;
        replacing these callbacks never resets confirmation expiry.
        """
        self._present = present
        self._is_answerable = is_answerable
        self._on_settled = on_settled

    def _answerable(self, session_id: str) -> bool:
        try:
            return bool(
                self._present
                and self._is_answerable
                and self._is_answerable(session_id)
            )
        except Exception:  # noqa: BLE001 - sanitized app/storage boundary
            return False

    def open_run(self, session_id: str, run_id: str) -> bool:
        """Register native run lifetime; never recreate closed run/session state."""
        identity = (session_id, run_id)
        if (
            self._closing
            or session_id in self._closed_sessions
            or identity in self._closed_runs
        ):
            return False
        self._sessions.add(session_id)
        self._runs.setdefault(identity, {})
        return True

    def _new_operation(
        self,
        session_id: str,
        target: ArchiveTarget | None,
        call_key: RecoveryCallKey | None,
    ) -> _Operation:
        loop = asyncio.get_running_loop()
        op = _Operation(
            secrets.token_urlsafe(24),
            session_id,
            target,
            call_key,
            loop.create_future(),
            loop.create_future(),
        )
        self._operations[op.operation_id] = op
        self._sessions.add(session_id)
        if call_key:
            self._runs[(session_id, call_key.run_id)][call_key.call_id] = (
                call_key.argument_digest,
                op.operation_id,
            )
        return op

    def _snapshot(self, op: _Operation) -> RecoveryOperationSnapshot:
        return RecoveryOperationSnapshot(
            op.operation_id,
            op.session_id,
            op.phase,
            # A conflict retains consumed presentation data for Restore as only.
            op.confirmation
            if op.phase in ("awaiting_confirmation", "settled")
            else None,
            op.result.result() if op.result.done() else None,
            op.admitted,
        )

    def snapshot(self, session_id: str) -> RecoveryOperationSnapshot | None:
        """Return trusted active state; also enforce expiry with the injected clock."""
        self.expire_pending()
        op = self._active.get(session_id)
        return self._snapshot(op) if op else None

    def receipts(self, session_id: str) -> tuple[RecoveryOperationSnapshot, ...]:
        """Return the bounded completion presentation cache, oldest first."""
        return tuple(
            self._snapshot(self._operations[oid])
            for oid in self._receipts.get(session_id, ())
        )

    def present_pending(self, session_id: str) -> bool:
        """Remount exact pending controls without minting new authorization."""
        state = self.snapshot(session_id)
        if not state or not state.confirmation or not self._answerable(session_id):
            return False
        try:
            return bool(self._present(state.confirmation))
        except Exception:  # noqa: BLE001 - sanitized app/storage boundary
            return False

    def _settle(self, op: _Operation, outcome: RecoveryOutcome) -> None:
        if op.result.done():
            return
        if op.timer:
            op.timer.cancel()
            op.timer = None
        op.phase = "settled"
        op.result.set_result(outcome)
        if not op.ready.done():
            op.ready.set_result(outcome)
        op.ready = op.result
        if self._active.get(op.session_id) is op:
            del self._active[op.session_id]
        # Only a name-conflict receipt needs its exact target for Restore as.
        if outcome.status != "name_conflict":
            op.target = None
            op.confirmation = None
        receipts = self._receipts.setdefault(op.session_id, OrderedDict())
        receipts[op.operation_id] = None
        while len(receipts) > MAX_SESSION_RECEIPTS:
            old_id, _ = receipts.popitem(last=False)
            old = self._operations.pop(old_id)
            for request_id in old.request_ids:
                self._requests.pop(request_id, None)
        if self._on_settled:
            try:
                self._on_settled(self._snapshot(op))
            except Exception:  # noqa: BLE001 - sanitized app/storage boundary
                return  # Presentation never changes a durable outcome.

    def expire_pending(self) -> None:
        """Expire by creation time; callable by deterministic clock tests/remount."""
        for op in tuple(self._active.values()):
            card = op.confirmation
            if (
                op.phase == "awaiting_confirmation"
                and card
                and card.expires_at <= self._monotonic()
            ):
                self._settle(
                    op, recovery_outcome("expired", op.target, next_action="refresh")
                )

    def _arm_expiry(self, op: _Operation) -> None:
        if op.timer:
            op.timer.cancel()

        def expire() -> None:
            self.expire_pending()
            if not op.result.done() and op.phase == "awaiting_confirmation":
                self._arm_expiry(op)

        delay = max(0, op.confirmation.expires_at - self._monotonic())
        op.timer = asyncio.get_running_loop().call_later(delay, expire)

    def _publish(
        self, op: _Operation, scope: RestoreScope | None, replacement_name: str | None
    ) -> RecoveryConfirmation | RecoveryOutcome:
        target = op.target
        card = RecoveryConfirmation(
            secrets.token_urlsafe(24),
            op.operation_id,
            op.session_id,
            target,
            allowed_restore_scopes(
                target.conversation_archived,
                bool(target.workspace and target.workspace.archived),
            ),
            scope,
            replacement_name,
            self._monotonic() + CONFIRMATION_TTL_SECONDS,
        )
        op.confirmation = card
        op.request_ids.add(card.request_id)
        self._requests[card.request_id] = op.operation_id
        if not self.present_pending(op.session_id):
            self._settle(op, recovery_outcome("unavailable", target))
            return op.result.result()
        self._arm_expiry(op)
        if op.ready.done():
            op.ready = asyncio.get_running_loop().create_future()
        op.ready.set_result(card)
        return card

    async def request(
        self,
        session_id: str,
        result_ref: str,
        *,
        call_key: RecoveryCallKey | None,
        scope: RestoreScope | None,
    ) -> RecoveryConfirmation | RecoveryOutcome:
        """Select immutable evidence and request UI confirmation, never approval."""
        async with self._lock:
            self.expire_pending()
            if self._closing or session_id in self._closed_sessions:
                return recovery_outcome("expired")
            calls = None
            if call_key:
                calls = self._runs.get((session_id, call_key.run_id))
                if (
                    call_key.session_id != session_id
                    or calls is None
                    or (session_id, call_key.run_id) in self._closed_runs
                ):
                    return recovery_outcome("expired")
                previous = calls.get(call_key.call_id)
                if previous:
                    digest, operation_id = previous
                    if digest != call_key.argument_digest:
                        return recovery_outcome("stale")
                    op = self._operations.get(operation_id)
                    if op is None:
                        return recovery_outcome("expired")
                    if op.result.done():
                        return op.result.result()
                    if op.phase == "awaiting_confirmation" and op.confirmation:
                        return op.confirmation
                    future = (
                        op.ready if op.phase == "awaiting_confirmation" else op.result
                    )
                    return_value = (op, future)
                else:
                    return_value = None
            else:
                return_value = None
            if return_value is None:
                try:
                    target = self._results.resolve(
                        session_id, result_ref, for_agent=call_key is not None
                    ).target
                except ArchiveRecoveryError as exc:
                    op = self._new_operation(session_id, None, call_key)
                    self._settle(op, recovery_outcome(exc.code, next_action="refresh"))
                    return op.result.result()
                active = self._active.get(session_id)
                if active:
                    card = active.confirmation
                    compatible = scope is None or (
                        card is not None
                        and (
                            scope == card.selected_scope
                            or (
                                card.selected_scope is None
                                and scope in card.allowed_scopes
                            )
                        )
                    )
                    if call_key is None and active.target == target and compatible:
                        if (
                            active.phase == "awaiting_confirmation"
                            and active.confirmation
                        ):
                            return active.confirmation
                        return_value = (
                            active,
                            active.ready
                            if active.phase == "awaiting_confirmation"
                            else active.result,
                        )
                    else:
                        op = self._new_operation(session_id, target, call_key)
                        self._settle(op, recovery_outcome("busy", target))
                        return op.result.result()
                else:
                    op = self._new_operation(session_id, target, call_key)
                    self._active[session_id] = op
                    self._start(op, self._prepare(op, scope))
                    return_value = (op, op.ready)
        op, future = return_value
        return await self._wait(op, future)

    def _start(self, op: _Operation, coroutine: Any) -> None:
        op.task = asyncio.create_task(coroutine)
        self._tasks[op.task] = op
        op.task.add_done_callback(self._forget_task)

    def _forget_task(self, task: asyncio.Task) -> None:
        self._tasks.pop(task, None)
        if not task.cancelled():
            task.exception()  # A cancelled read waiter still leaves an observed result.

    async def _prepare(self, op: _Operation, scope: RestoreScope | None) -> None:
        try:
            current = await read_recovery_target(self._app, op.target.conversation_id)
            if op.result.done():
                return
            if current != op.target:
                self._settle(
                    op, recovery_outcome("stale", op.target, next_action="refresh")
                )
                return
            # Only validated current metadata supplies the confirmation label.
            # ID/version/ownership and the complete workspace snapshot matched.
            op.target = current
            choices = allowed_restore_scopes(
                current.conversation_archived,
                bool(current.workspace and current.workspace.archived),
            )
            if not choices:
                self._settle(
                    op,
                    recovery_outcome(
                        "already_available",
                        current,
                        chat_available=True,
                        next_action="open",
                    ),
                )
            elif scope is not None and scope not in choices:
                self._settle(op, recovery_outcome("unavailable", current))
            elif op.cancel_event.is_set():
                self._settle(op, recovery_outcome("cancelled", current))
            else:
                self._publish(op, scope, None)
        except ArchiveRecoveryError as exc:
            self._settle(
                op, recovery_outcome(exc.code, op.target, next_action="refresh")
            )
        except Exception:  # noqa: BLE001 - sanitized app/storage boundary
            self._settle(
                op, recovery_outcome("storage_error", op.target, next_action="refresh")
            )

    async def _wait(self, op: _Operation, future: asyncio.Future) -> Any:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._cancel(op)
            raise

    def _cancel(self, op: _Operation) -> None:
        op.cancel_event.set()
        if op.phase == "awaiting_confirmation":
            self._settle(op, recovery_outcome("cancelled", op.target))

    async def decide(
        self,
        session_id: str,
        request_id: str,
        *,
        allow: bool,
        scope: RestoreScope | None,
    ) -> RecoveryOutcome:
        """Consume the exact displayed request once; only app UI may call this."""
        async with self._lock:
            self.expire_pending()
            op = self._operations.get(self._requests.get(request_id, ""))
            if op is None or op.session_id != session_id:
                return recovery_outcome("expired")
            if not op.result.done() and op.phase == "awaiting_confirmation":
                card = op.confirmation
                if not card or card.request_id != request_id:
                    return recovery_outcome("expired")
                selected = scope or card.selected_scope
                if not allow:
                    self._settle(op, recovery_outcome("declined", op.target))
                elif (
                    self._closing
                    or session_id in self._closed_sessions
                    or not self._answerable(session_id)
                ) or selected not in card.allowed_scopes:
                    self._settle(op, recovery_outcome("unavailable", op.target))
                else:
                    # No awaits from authorization consumption to retained task ownership.
                    op.admitted = True
                    op.phase = "admitted"
                    op.confirmation = replace(card, selected_scope=selected)
                    if op.timer:
                        op.timer.cancel()
                        op.timer = None
                    self._start(op, self._execute(op, selected))
        return await self._wait(op, op.result)

    async def _execute(self, op: _Operation, scope: RestoreScope) -> None:
        def phase(value: RecoveryPhase) -> None:
            if (
                op.phase == "admitted"
                and value in ("workspace_write", "chat_write")
                and not self._answerable(op.session_id)
            ):
                raise ArchiveRecoveryError(
                    "unavailable", "The confirmation UI is unavailable."
                )
            op.phase = value

        try:
            result = await execute_restore(
                self._app,
                op.target,
                scope,
                replacement_name=op.confirmation.replacement_name,
                cancel_event=op.cancel_event,
                on_phase=phase,
            )
        except Exception:  # noqa: BLE001 - sanitized app/storage boundary
            result = recovery_outcome("storage_error", op.target, next_action="refresh")
        self._settle(op, result)

    async def revise_workspace_name(
        self, session_id: str, request_id: str, name: str
    ) -> RecoveryConfirmation:
        """Revalidate, invalidate the old card, and confirm the exact new name."""
        name = validate_workspace_name(name)
        async with self._lock:
            old = self._name_revision_owner(session_id, request_id)
            target = old.target
        try:
            read_task = asyncio.create_task(
                read_recovery_target(self._app, target.conversation_id)
            )
            self._tasks[read_task] = old
            read_task.add_done_callback(self._forget_task)
            current = await asyncio.shield(read_task)
            if current != target:
                raise ArchiveRecoveryError(
                    "stale", "The selected target changed; refresh recovery."
                )
        except Exception as exc:  # noqa: BLE001 - sanitized app/storage boundary
            status = (
                exc.code if isinstance(exc, ArchiveRecoveryError) else "storage_error"
            )
            async with self._lock:
                if (
                    self._active.get(session_id) is old
                    and old.phase == "awaiting_confirmation"
                ):
                    self._settle(
                        old, recovery_outcome(status, target, next_action="refresh")
                    )
            raise ArchiveRecoveryError(
                status, "Refresh recovery before choosing a name."
            ) from None
        async with self._lock:
            old = self._name_revision_owner(session_id, request_id)
            if old.result.done():
                if session_id in self._active:
                    raise ArchiveRecoveryError(
                        "busy", "Another recovery request is active."
                    )
                op = self._new_operation(session_id, target, None)
                self._active[session_id] = op
            else:
                op = old
                self._requests.pop(request_id, None)
                op.request_ids.discard(request_id)
            scope = old.confirmation.selected_scope
            result = self._publish(op, scope, name)
            if isinstance(result, RecoveryOutcome):
                raise ArchiveRecoveryError(result.status, result.message)
            return result

    def _name_revision_owner(self, session_id: str, request_id: str) -> _Operation:
        self.expire_pending()
        old = self._operations.get(self._requests.get(request_id, ""))
        if (
            self._closing
            or session_id in self._closed_sessions
            or old is None
            or old.session_id != session_id
            or old.confirmation is None
            or old.confirmation.request_id != request_id
        ):
            raise ArchiveRecoveryError(
                "expired", "Refresh recovery before choosing a name."
            )
        if old.phase != "awaiting_confirmation" and (
            not old.result.done() or old.result.result().status != "name_conflict"
        ):
            raise ArchiveRecoveryError("expired", "This decision was already consumed.")
        if not old.target.workspace or not old.target.workspace.archived:
            raise ArchiveRecoveryError(
                "unavailable", "The workspace does not need restoration."
            )
        return old

    async def wait_result(self, operation_id: str) -> RecoveryOutcome:
        """Join coordinator custody; cancellation stops subsequent unstarted writes."""
        self.expire_pending()
        op = self._operations.get(operation_id)
        if op is None:
            return recovery_outcome("expired")
        return await self._wait(op, op.result)

    @staticmethod
    async def _join_before_release(task: asyncio.Task) -> None:
        # Even cancellation of the runtime's close waiter must not let its
        # finally block release database handles underneath a started thread.
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def close_run(self, session_id: str, run_id: str) -> None:
        """Reject late callbacks and join this run's active recovery work."""
        identity = (session_id, run_id)
        self._closed_runs.add(identity)
        op = self._active.get(session_id)
        if op and op.call_key and op.call_key.run_id == run_id:
            self._cancel(op)

        async def finish() -> None:
            tasks = [
                task
                for task, owner in self._tasks.items()
                if owner.session_id == session_id
                and owner.call_key
                and owner.call_key.run_id == run_id
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._runs.pop(identity, None)

        await self._join_before_release(asyncio.create_task(finish()))

    async def close_session(self, session_id: str) -> None:
        """Refuse new requests, settle work, then release ephemeral session state."""
        self._closed_sessions.add(session_id)
        op = self._active.get(session_id)
        if op:
            self._cancel(op)
        task = self._session_closures.get(session_id)
        if task is None:
            task = asyncio.create_task(self._finish_session(session_id))
            self._session_closures[session_id] = task
        await self._join_before_release(task)

    async def _finish_session(self, session_id: str) -> None:
        tasks = [
            task
            for task, owner in self._tasks.items()
            if owner.session_id == session_id
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._results.close_session(session_id)
        for identity in tuple(self._runs):
            if identity[0] == session_id:
                self._runs.pop(identity)
        for oid in self._receipts.pop(session_id, ()):
            old = self._operations.pop(oid, None)
            if old:
                for request_id in old.request_ids:
                    self._requests.pop(request_id, None)
        self._closed_runs = {
            identity for identity in self._closed_runs if identity[0] != session_id
        }

    async def shutdown(self) -> None:
        """Join all started storage before the runtime releases database handles."""
        self._closing = True
        for op in tuple(self._active.values()):
            self._cancel(op)
        if self._shutdown_task is None:

            async def finish() -> None:
                await asyncio.gather(
                    *(self.close_session(sid) for sid in self._sessions)
                )

            self._shutdown_task = asyncio.create_task(finish())
        await self._join_before_release(self._shutdown_task)
