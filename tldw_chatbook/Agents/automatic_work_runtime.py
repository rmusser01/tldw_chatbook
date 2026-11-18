"""Private accepted-attempt context shared by native Console generations."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
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


def _owned_goal_call(function):
    """Retire Goal checks on each physical caller; ordinary automatic work is unchanged."""
    from functools import wraps

    @wraps(function)
    def invoke(context, *args, **kwargs):
        if context.goal is None:
            return function(context, *args, **kwargs)
        from .goal_storage_lifetime import owned_goal_storage

        with owned_goal_storage(context.goal._coordinator.service):
            return function(context, *args, **kwargs)

    return invoke


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
    attempt_kind: Literal["wake", "chat_start", "goal_iteration", "workflow_phase"] = (
        "wake"
    )
    goal: GoalIterationAuthorization | None = field(
        default=None, repr=False, compare=False
    )
    workflow: object | None = field(default=None, repr=False, compare=False)
    _accepted: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False, compare=False
    )

    @contextmanager
    def _bound_scope(self) -> Iterator[AutomaticWorkContext]:
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    @contextmanager
    def scope(self) -> Iterator[AutomaticWorkContext]:
        """Bind this authority for helper calls and restore the caller's scope."""
        with self._bound_scope():
            if self.goal is None:
                yield self
            else:
                from .goal_storage_lifetime import owned_goal_storage

                with owned_goal_storage(self.goal._coordinator.service):
                    yield self

    def _read_attempt(self):
        readers = {
            "wake": self.ledger.read_attempt,
            "chat_start": self.ledger.read_chat_start_attempt,
            "goal_iteration": self.ledger.read_goal_attempt,
            "workflow_phase": self.ledger.read_workflow_attempt,
        }
        if self.attempt_kind not in readers:
            raise AutomaticWorkRefused("acceptance_required")
        return readers[self.attempt_kind](self.attempt_id, owner_id=self.owner_id)

    def _accepted_states(self):
        return (
            {"accepted"}
            if self.attempt_kind in {"goal_iteration", "workflow_phase"}
            else {"accepted", "completed"}
        )

    @_owned_goal_call
    def policy_kind(self) -> str:
        """Policy follows the immutable allowance root, including child chats."""
        return self.ledger.policy_kind(self.chain_id)

    @_owned_goal_call
    def mark_accepted(self) -> None:
        """Latch acceptance only after the required durable fence succeeded."""
        if self.attempt_kind == "workflow_phase":
            from .workflow_phase import WorkflowPhaseAuthorization

            if (
                type(self.workflow) is not WorkflowPhaseAuthorization
                or self.workflow.context is not self
            ):
                raise AutomaticWorkRefused("acceptance_required")
            self.workflow.check()
        attempt = self._read_attempt()
        if (
            attempt.chain_id != self.chain_id
            or attempt.state not in self._accepted_states()
        ):
            raise AutomaticWorkRefused("acceptance_required")
        self._accepted.set()

    def _workflow_dispatch(self):
        if self.workflow is None:
            return None
        from .workflow_task_scope import current_task_authorization

        task = current_task_authorization()
        if task is None:
            return None
        if task.phase is not self.workflow or task.phase.context is not self:
            raise AutomaticWorkRefused("workflow_authority_changed")
        return self.workflow._task_dispatch(task)

    @_owned_goal_call
    def check(self) -> AutomaticWorkSnapshot:
        """Enforce accepted authority, the root kill switch, and live limits."""
        from .agent_service import coerce_autowake_enabled

        if not self._accepted.is_set():
            raise AutomaticWorkRefused("acceptance_required")
        dispatch = None
        if self.attempt_kind == "workflow_phase":
            from .workflow_phase import WorkflowPhaseAuthorization

            if (
                type(self.workflow) is not WorkflowPhaseAuthorization
                or self.workflow.context is not self
            ):
                raise AutomaticWorkRefused("acceptance_required")
            dispatch = self._workflow_dispatch()
            if dispatch is None:
                self.workflow.check()
            else:
                with self.ledger._db.connection() as conn:
                    self.ledger._workflow_current(
                        conn,
                        self.attempt_id,
                        self.owner_id,
                        accepted=True,
                        _dispatch=dispatch,
                    )
        attempt = self._read_attempt()
        if (
            attempt.chain_id != self.chain_id
            or attempt.state not in self._accepted_states()
        ):
            raise AutomaticWorkRefused("review_required")
        policy = self.policy_kind()
        is_goal = policy == "goal_iteration"
        is_workflow = policy == "workflow_phase"
        setting = (
            "workflow_runs_enabled"
            if is_workflow
            else "goal_runs_enabled"
            if is_goal
            else "autowake_enabled"
        )
        reason = (
            "workflow_runs_disabled"
            if is_workflow
            else "goal_runs_disabled"
            if is_goal
            else "autowake_disabled"
        )
        if not coerce_autowake_enabled(
            _setting(setting, not (is_goal or is_workflow))
        ):
            self.ledger.pause(self.chain_id, reason)
            raise AutomaticWorkRefused(reason)
        if self.goal is not None:
            self.goal.check_binding()
        limits = AutomaticWorkLimits.from_settings(self.policy_kind())
        if (is_goal or is_workflow) and limits.output_tokens <= 0:
            self.ledger.pause(self.chain_id, "output_tokens_budget")
            raise AutomaticWorkRefused("output_tokens_budget")
        return self.ledger.check_active(
            self.chain_id,
            owner_id=self.owner_id,
            limits=limits,
            _dispatch=dispatch if self.workflow is not None else None,
        )

    def should_cancel(self) -> bool:
        """Fail closed through existing cooperative cancellation probes."""
        try:
            self.check()
        except Exception:  # noqa: BLE001 - cancellation must fail closed on ledger failure
            return True
        return False

    @_owned_goal_call
    def output_cap(self, requested: int | None) -> int:
        """Narrow an ordinary output limit without replenishing chain limits."""
        snapshot = self.check()
        cap = min(
            snapshot.limits.output_tokens,
            AutomaticWorkLimits.from_settings(self.policy_kind()).output_tokens,
        )
        if type(requested) is int and requested > 0:
            cap = min(cap, requested)
        if cap <= 0:
            self.ledger.pause(self.chain_id, "output_tokens_budget")
            raise AutomaticWorkRefused("output_tokens_budget")
        return cap

    @_owned_goal_call
    def begin_call(self, input_tokens: int, output_tokens: int) -> str:
        """Durably admit one physical provider generation and its token budget."""
        self.check()
        reservation = self.ledger.admit_call(
            self.chain_id,
            call_id=uuid4().hex,
            owner_id=self.owner_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            limits=AutomaticWorkLimits.from_settings(self.policy_kind()),
            _dispatch=self._workflow_dispatch(),
        )
        if reservation is None:
            raise AutomaticWorkRefused("call_already_dispatched")
        return reservation

    @_owned_goal_call
    def settle_call(self, reservation_id: str, actual_tokens: int | None) -> None:
        """Keep missing or interrupted usage conservative and durable."""
        self.ledger.settle(
            reservation_id, owner_id=self.owner_id, actual_amount=actual_tokens
        )
