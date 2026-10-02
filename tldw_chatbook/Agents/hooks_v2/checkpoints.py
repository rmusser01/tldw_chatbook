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
    owner_id: str
    retirement_owner_id: str
    current: Callable
    stage_context: Callable
    retain_context: bool
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
        # Settled failed operations need only handler IDs, never event bodies
        # or callbacks. Aggregate owning/dependent failures per live gate owner.
        self._failures: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
        self._contexts: dict[str, list[tuple[HookEvent, HookEventOutcome]]] = {}
        self._closed: set[str] = set()
        self._parents: dict[str, str | None] = {}

    def bind_owner(self, owner_id: str, parent_id: str | None = None) -> None:
        """Bind a host scope once; reject unknown parents, closure and reparenting."""
        with self._condition:
            if owner_id in self._closed or owner_id == parent_id:
                raise HookCheckpointError("hook owner closed or cyclic")
            if parent_id is not None:
                self.owners(parent_id)
                if owner_id in self.owners(parent_id):
                    raise HookCheckpointError("hook owner cyclic")
            if owner_id in self._parents and self._parents[owner_id] != parent_id:
                raise HookCheckpointError("hook owner cannot be reparented")
            self._parents[owner_id] = parent_id

    def owners(self, owner_id: str) -> tuple[str, ...]:
        """Return the exact live scope and its fixed host ancestors."""
        with self._condition:
            result = []
            current = owner_id
            while current is not None:
                if current in self._closed or current not in self._parents:
                    raise HookCheckpointError("hook owner closed or unknown")
                result.append(current)
                current = self._parents[current]
            return tuple(result)

    def is_current(self, owner_id: str) -> bool:
        try:
            self.owners(owner_id)
            return True
        except HookCheckpointError:
            return False

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
        owner_id: str | None = None,
        current: Callable | None = None,
        stage_context: Callable | None = None,
        retain_context: bool = True,
        retirement_owner_id: str | None = None,
    ) -> str:
        """Install before publishing completion; never reuse an event token."""
        owner = owner_id or self._owner(event)
        with self._condition:
            if owner_id is None and owner not in self._parents:
                self.bind_owner(owner)
            self.owners(owner)
            retirement_owner = retirement_owner_id or owner
            if owner not in self.owners(retirement_owner):
                raise HookCheckpointError("checkpoint retirement owner outside gate")
            if owner in self._closed:
                raise HookCheckpointError("hook owner closed")
            if any(
                item.event.event_id == event.event_id for item in self._entries.values()
            ):
                raise HookCheckpointError("hook event already registered")
            token = uuid.uuid4().hex
            self._entries[token] = _Checkpoint(
                event,
                owner,
                retirement_owner,
                current or self._current,
                stage_context or self._stage_context,
                retain_context,
                frozenset(requirements),
                frozenset(dependency_requirements),
            )
            return token

    def accept(self, token: str, result: HookEventOutcome) -> None:
        """Atomically accept current effects and settle their exact requirements."""
        with self._condition:
            entry = self._entries[token]
            if not entry.pending:
                raise HookCheckpointError("hook checkpoint already settled")
            owner = entry.owner_id
            try:
                current = (
                    self.is_current(owner)
                    and self.is_current(entry.retirement_owner_id)
                    and entry.current(entry.event)
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
                        entry.stage_context(entry.event, accepted_outcome)
                    except Exception:  # noqa: BLE001 -- fail closed
                        entry.failed = required
                        entry.reason = "hook context acceptance failed"
                    else:
                        if entry.retain_context:
                            self._contexts.setdefault(owner, []).append(
                                (entry.event, accepted_outcome)
                            )
            entry.pending = False
            if entry.retirement_owner_id in self._closed:
                self._retire_checkpoint(token)
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
            if entry.retirement_owner_id in self._closed:
                self._retire_checkpoint(token)
            self._condition.notify_all()

    def _check(self, owner_id, required_handler_ids, terminal):
        owners = self.owners(owner_id)
        if required_handler_ids is None:
            raise HookCheckpointError("hook dependency mapping unknown")
        selected = frozenset(required_handler_ids)
        pending = False
        failed = any(
            owning or dependent & selected
            for owner, (owning, dependent) in self._failures.items()
            if owner in owners
        )
        for entry in self._entries.values():
            if entry.owner_id not in owners:
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

    def assert_continuation_dependencies(
        self, parent_id: str, *, required_handler_ids: tuple[str, ...] | None
    ) -> None:
        """Check a nested operation's actual prerequisites in its live ancestry.

        Work fulfilling an event is not the parent's next model input. It may
        run while unrelated owning-event barriers remain pending, but cannot
        ignore an owning OR dependent requirement named by its resolved tool.
        This query never removes, settles or reparents the outer barriers.
        """
        with self._condition:
            owners = self.owners(parent_id)
            if required_handler_ids is None:
                raise HookCheckpointError("hook dependency mapping unknown")
            selected = frozenset(required_handler_ids)
            failed = any(
                (owning | dependent) & selected
                for owner, (owning, dependent) in self._failures.items()
                if owner in owners
            )
            pending = False
            for entry in self._entries.values():
                if entry.owner_id not in owners:
                    continue
                relevant = (entry.requirements | entry.dependencies) & selected
                pending |= bool(entry.pending and relevant)
                failed |= bool(entry.failed & relevant)
            if failed:
                raise HookCheckpointError("required hook checkpoint failed")
            if pending:
                raise HookCheckpointError("required hook checkpoint pending")

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
            self._failures.pop(owner_id, None)
            self._condition.notify_all()

    def _retire_checkpoint(self, token: str) -> None:
        """Discard settled operation closures, preserving only scoped failures."""
        entry = self._entries[token]
        if entry.pending:
            return
        if entry.failed and self.is_current(entry.owner_id):
            owning, dependent = self._failures.get(
                entry.owner_id, (frozenset(), frozenset())
            )
            self._failures[entry.owner_id] = (
                owning | (entry.failed & entry.requirements),
                dependent | (entry.failed & entry.dependencies),
            )
        self._entries.pop(token)

    def retire_owner(self, owner_id: str) -> None:
        """Drop terminal scope state once no live descendants retain it."""
        with self._condition:
            self.close_owner(owner_id)
            if any(parent == owner_id for parent in self._parents.values()):
                return
            for token, entry in tuple(self._entries.items()):
                if entry.retirement_owner_id == owner_id:
                    self._retire_checkpoint(token)
            parent = self._parents.pop(owner_id, None)
            # Runtime-bounded identity tombstones prevent a late owner from
            # being rebound. Bodies and completed checkpoints are discarded.
            if parent in self._closed:
                self.retire_owner(parent)
