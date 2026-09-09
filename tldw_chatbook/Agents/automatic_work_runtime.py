"""Private accepted-wake context shared by native Console generations."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from .automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkRefused,
    AutomaticWorkSnapshot,
)
from .run_log import _setting

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_goal_runs import GoalIterationAuthorization
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger


_CURRENT: ContextVar[AutomaticWorkContext | None] = ContextVar(
    "automatic_work", default=None
)


def current_automatic_work() -> AutomaticWorkContext | None:
    """Return trusted execution context, never deriving origin from payloads."""
    return _CURRENT.get()


@contextmanager
def manual_work_scope() -> Iterator[None]:
    """Clear inherited automatic authority for an explicitly manual submission."""
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@dataclass(frozen=True)
class AutomaticWorkContext:
    """Immutable chain authority with a process-local acceptance latch.

    A coordinator calls mark_accepted only after winning the durable fence.
    Explicitly carry this same object into owned child and tool threads.
    """

    ledger: AutomaticWorkLedger = field(repr=False)
    chain_id: str
    owner_id: str
    attempt_id: str
    attempt_kind: str = "fleet_wake"
    goal: GoalIterationAuthorization | None = field(
        default=None, repr=False, compare=False
    )
    _accepted: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False, compare=False
    )

    @contextmanager
    def scope(self) -> Iterator[AutomaticWorkContext]:
        """Bind this authority for helper calls and restore the caller's scope."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def _read_attempt(self):
        if self.attempt_kind == "goal_iteration":
            return self.ledger.read_goal_attempt(
                self.attempt_id, owner_id=self.owner_id
            )
        if self.attempt_kind != "fleet_wake":
            raise AutomaticWorkRefused("attempt_kind_unknown")
        return self.ledger.read_attempt(self.attempt_id, owner_id=self.owner_id)

    def _accepted_states(self):
        return (
            {"accepted"}
            if self.attempt_kind == "goal_iteration"
            else {"accepted", "completed"}
        )

    def mark_accepted(self) -> None:
        """Latch acceptance only after the required durable fence succeeded."""
        attempt = self._read_attempt()
        if (
            attempt.chain_id != self.chain_id
            or attempt.state not in self._accepted_states()
        ):
            raise AutomaticWorkRefused("acceptance_required")
        self._accepted.set()

    def check(self) -> AutomaticWorkSnapshot:
        """Enforce accepted authority, the kill switch, and live finite limits."""
        from .agent_service import _coerce_autowake_enabled

        if not self._accepted.is_set():
            raise AutomaticWorkRefused("acceptance_required")
        attempt = self._read_attempt()
        if (
            attempt.chain_id != self.chain_id
            or attempt.state not in self._accepted_states()
        ):
            raise AutomaticWorkRefused("review_required")
        goal = self.attempt_kind == "goal_iteration"
        setting = "goal_runs_enabled" if goal else "autowake_enabled"
        reason = "goal_runs_disabled" if goal else "autowake_disabled"
        if not _coerce_autowake_enabled(_setting(setting, not goal)):
            self.ledger.pause(self.chain_id, reason)
            raise AutomaticWorkRefused(reason)
        if self.goal is not None:
            self.goal.check_binding()
        limits = AutomaticWorkLimits.from_settings(self.attempt_kind)
        if goal and limits.output_tokens <= 0:
            self.ledger.pause(self.chain_id, "output_tokens_budget")
            raise AutomaticWorkRefused("output_tokens_budget")
        return self.ledger.check_active(
            self.chain_id,
            owner_id=self.owner_id,
            limits=limits,
        )

    def should_cancel(self) -> bool:
        """Fail closed through existing cooperative cancellation probes."""
        try:
            self.check()
        except Exception:  # noqa: BLE001 - cancellation must fail closed on ledger failure
            return True
        return False

    def output_cap(self, requested: int | None) -> int:
        """Narrow an ordinary output limit without replenishing chain limits."""
        snapshot = self.check()
        cap = min(
            snapshot.limits.output_tokens,
            AutomaticWorkLimits.from_settings(self.attempt_kind).output_tokens,
        )
        if type(requested) is int and requested > 0:
            cap = min(cap, requested)
        if cap <= 0:
            self.ledger.pause(self.chain_id, "output_tokens_budget")
            raise AutomaticWorkRefused("output_tokens_budget")
        return cap

    def begin_call(self, input_tokens: int, output_tokens: int) -> str:
        """Durably admit one physical provider generation and its token budget."""
        self.check()
        reservation = self.ledger.admit_call(
            self.chain_id,
            call_id=uuid4().hex,
            owner_id=self.owner_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            limits=AutomaticWorkLimits.from_settings(self.attempt_kind),
        )
        if reservation is None:
            raise AutomaticWorkRefused("call_already_dispatched")
        return reservation

    def settle_call(self, reservation_id: str, actual_tokens: int | None) -> None:
        """Keep missing or interrupted usage conservative and durable."""
        self.ledger.settle(
            reservation_id, owner_id=self.owner_id, actual_amount=actual_tokens
        )
