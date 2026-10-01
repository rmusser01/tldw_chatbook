"""Run-owned post-event barriers; acceptance and admission share one lock."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from .engine import HookEventOutcome
from .models import HookEvent


class HookCheckpointError(RuntimeError):
    """The owning input needs remediation or cancellation before admission."""


@dataclass
class _Checkpoint:
    event: HookEvent
    requirements: frozenset[str]
    dependencies: frozenset[str]
    pending: bool = True
    failed: frozenset[str] = frozenset()
    reason: str = ""


class HookCheckpointStore:
    """One accepted turn's synchronized checkpoints and pending context lane.

    ``current`` is a synchronous host-currentness probe, never hook execution.
    The context lane carries validated outcomes until the existing model-history
    owner consumes them. It neither persists bodies nor owns plugin authority.
    """

    def __init__(
        self,
        *,
        current: Callable[[HookEvent], bool] = lambda _event: True,
        accept_current: Callable = lambda _event, _result: True,
        stage_context: Callable = lambda _event, _result: None,
    ):
        self._condition = threading.Condition(threading.RLock())
        self._current = current
        self._accept_current = accept_current
        self._stage_context = stage_context
        self._entries: dict[str, _Checkpoint] = {}
        self._contexts: dict[str, list[tuple[HookEvent, HookEventOutcome]]] = {}
        self._closed: set[str] = set()

    @staticmethod
    def _owner(event: HookEvent) -> str:
        if not event.run_id:
            raise ValueError("post checkpoint requires an exact run owner")
        return event.run_id

    def begin(
        self,
        event: HookEvent,
        requirements: tuple[str, ...],
        *,
        dependency_requirements: tuple[str, ...] = (),
    ) -> str:
        """Install before publishing completion; never reuse an event token."""
        owner = self._owner(event)
        with self._condition:
            if owner in self._closed:
                raise HookCheckpointError("hook owner closed")
            if any(
                item.event.event_id == event.event_id for item in self._entries.values()
            ):
                raise HookCheckpointError("hook event already registered")
            token = uuid.uuid4().hex
            self._entries[token] = _Checkpoint(
                event, frozenset(requirements), frozenset(dependency_requirements)
            )
            return token

    def accept(self, token: str, result: HookEventOutcome) -> None:
        """Atomically accept current effects and settle their exact requirements."""
        with self._condition:
            entry = self._entries[token]
            if not entry.pending:
                raise HookCheckpointError("hook checkpoint already settled")
            owner = self._owner(entry.event)
            try:
                current = (
                    owner not in self._closed
                    and self._current(entry.event)
                    and self._accept_current(entry.event, result)
                )
            except Exception:  # noqa: BLE001 -- host failure retains requirements
                current = False
            accepted = {handler_id for handler_id, _ in result.accepted}
            failures = {failure.handler_id for failure in result.failures}
            required = entry.requirements | entry.dependencies
            entry.failed = frozenset((required - accepted) | (required & failures))
            if not current or result.outstanding_cleanup:
                entry.failed = required
                entry.reason = "stale or cancelled hook result"
            elif not result.allowed:
                entry.failed |= entry.requirements
                entry.reason = "required hook failed"
            elif entry.failed:
                entry.reason = "required hook completion missing"
            # Failed dependency effects cannot satisfy that dependency. Preserve
            # successful independently scoped effects without widening failure.
            if (
                current
                and not (entry.failed & entry.requirements)
                and not result.outstanding_cleanup
            ):
                accepted_results = tuple(
                    (handler_id, value)
                    for handler_id, value in result.accepted
                    if handler_id not in entry.failed
                )
                if accepted_results:
                    accepted_outcome = HookEventOutcome(accepted=accepted_results)
                    try:
                        self._stage_context(entry.event, accepted_outcome)
                    except Exception:  # noqa: BLE001 -- fail closed
                        entry.failed = required
                        entry.reason = "hook context acceptance failed"
                    else:
                        self._contexts.setdefault(owner, []).append(
                            (entry.event, accepted_outcome)
                        )
            entry.pending = False
            self._condition.notify_all()

    def fail(self, token: str, reason: str) -> None:
        """Keep the failed requirement; never retry settled tool effects."""
        with self._condition:
            entry = self._entries[token]
            if not entry.pending:
                raise HookCheckpointError("hook checkpoint already settled")
            entry.pending = False
            entry.failed = entry.requirements | entry.dependencies
            entry.reason = reason
            self._condition.notify_all()

    def _check(self, owner_id, required_handler_ids, terminal):
        if owner_id in self._closed:
            raise HookCheckpointError("hook owner closed")
        if required_handler_ids is None:
            raise HookCheckpointError("hook dependency mapping unknown")
        selected = frozenset(required_handler_ids)
        pending = False
        failed = False
        for entry in self._entries.values():
            if self._owner(entry.event) != owner_id:
                continue
            relevant = entry.requirements | (entry.dependencies & selected)
            if entry.pending and (relevant or (terminal and entry.dependencies)):
                pending = True
            if entry.failed & relevant:
                failed = True
        # Terminal settlement joins every required event before reporting an
        # already-known failure; next-input refusal may fail immediately.
        if failed and not terminal:
            raise HookCheckpointError("required hook checkpoint failed")
        if pending:
            raise HookCheckpointError("required hook checkpoint pending")
        if failed:
            raise HookCheckpointError("required hook checkpoint failed")

    def assert_next_input_allowed(
        self, owner_id: str, *, required_handler_ids: tuple[str, ...] | None = ()
    ) -> None:
        """Check owning requirements plus positively resolved dependent use."""
        with self._condition:
            self._check(owner_id, required_handler_ids, False)

    def wait(
        self,
        owner_id: str,
        *,
        required_handler_ids: tuple[str, ...] | None = (),
        terminal: bool = False,
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> None:
        """Wait without holding the lock during external hook work."""
        with self._condition:
            while True:
                if should_cancel():
                    self.close_owner(owner_id)
                    raise HookCheckpointError("hook owner cancelled")
                try:
                    self._check(owner_id, required_handler_ids, terminal)
                    return
                except HookCheckpointError as error:
                    if str(error) != "required hook checkpoint pending":
                        raise
                self._condition.wait(0.05)

    def drain_context(
        self, owner_id: str
    ) -> tuple[tuple[HookEvent, HookEventOutcome], ...]:
        """Transfer each accepted whole contribution to its model owner once."""
        with self._condition:
            return tuple(self._contexts.pop(owner_id, ()))

    def close_owner(self, owner_id: str) -> None:
        """Seal immediately; a late result cannot reopen or retarget this owner."""
        with self._condition:
            self._closed.add(owner_id)
            self._contexts.pop(owner_id, None)
            self._condition.notify_all()
