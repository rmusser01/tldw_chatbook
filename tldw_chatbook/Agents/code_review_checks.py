"""Per-review check facts; storage, permission and execution owners remain external."""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .code_review_models import (
    CheckProposal,
    CheckReceipt,
    ReviewRefused,
    ReviewRunBinding,
)


@dataclass(frozen=True)
class ReviewCheckNotice:
    review_id: str
    proposal_id: str
    child_run_id: str


@dataclass(frozen=True)
class ReviewCheckClaim:
    claim_id: str
    review_id: str
    proposal_id: str
    verifier_attempt_id: str
    primary_run_id: str
    target_digest: str
    tool_name: str | None = None
    arguments: dict | None = None
    receipt: CheckReceipt | None = None


@dataclass
class PendingCheck:
    proposal_id: str
    proposal: CheckProposal
    binding: ReviewRunBinding
    verifier_run_id: str
    primary_run_id: str
    state: str = "pending"
    claim: ReviewCheckClaim | None = None
    receipt: CheckReceipt | None = None
    native_execution_receipt_id: str | None = None
    execution: object | None = field(default=None, repr=False)
    result_read: bool = False
    deadline: float | None = None
    wait_count: int = 0
    notice_count: int = 0
    transport_reason: str | None = None
    result_reservation: object | None = field(default=None, repr=False)
    changed: threading.Event = field(default_factory=threading.Event, repr=False)

    def result(self, outcome: str, evidence: str) -> CheckReceipt:
        return CheckReceipt(
            schema_version=1,
            receipt_id=uuid.uuid4().hex,
            proposal_id=self.proposal_id,
            review_id=self.binding.review_id,
            verifier_attempt_id=self.binding.attempt_id,
            target_digest=self.binding.target_digest,
            outcome=outcome,
            evidence=evidence,
            native_execution_receipt_id=self.native_execution_receipt_id,
        )


_ACTIVE_EXECUTION = ContextVar("review_check_execution", default=None)
CHECK_GENERATED_BYTES = 1024 * 1024
CHECK_GENERATED_FILES = 128
CHECK_OUTPUT_BYTES = 16384


@dataclass
class ReviewCheckExecution:
    """A claimed check's existing native process/file ownership, never wire data.

    ``boundary`` comes only from the actual executor's reserve_review_boundary
    factory. That executor must enforce aggregate bytes/files, including copies,
    before it materializes anything. Stock Raw CLI supplies no such factory.
    """

    service: object
    provider: object
    check: PendingCheck
    boundary: object
    reservation: object
    arguments: dict
    accepted_id: str | None = None
    native_result: object | None = None
    registered_result: object | None = None
    started: bool = False
    released: bool = False

    @contextmanager
    def activate(self):
        with self.service.store._tab.lock:
            context = self.service.store._context(self.check.binding.review_id)
            self.service.store._live(context)
            if self.started or self.released or self.check.receipt is not None:
                raise ReviewRefused("check_execution_already_accepted")
            self.started = True
        token = _ACTIVE_EXECUTION.set(self)
        try:
            yield
        finally:
            _ACTIVE_EXECUTION.reset(token)
            with self.service.store._tab.lock:
                if self.accepted_id is None:
                    try:
                        if (
                            not self.service.store._tab.closed.is_set()
                            and self.check.receipt is None
                        ):
                            self.service._resolve_check(
                                context,
                                self.check,
                                "unavailable",
                                "process_not_started",
                            )
                    finally:
                        if not self.released:
                            self.release()

    @contextmanager
    def admission_scope(self):
        """Reject fenced admission; retain native ownership once admitted.

        Close publishes its fence without this lock. An admission already inside
        this scope remains in flight until the native commit/settlement proves its
        outcome; only the native accepted callback creates an accepted receipt.
        Source/UI validation occurs before taking the publication lock.
        """
        from .run_context import use_run_actor

        with self.service.source_authority.scope():
            with use_run_actor(self.service.actor), self.service.store._tab.lock:
                context = self.service.store._context(self.check.binding.review_id)
                self.service.store._live(context)
                self.service._bound(context)
                stream = context.streams[
                    (self.check.binding.slot_id, self.check.binding.attempt_id)
                ]
                if (
                    self.check.receipt is not None
                    or stream.terminal_status is not None
                    or self.released
                    or self.accepted_id is not None
                    or time.monotonic()
                    >= min(self.check.deadline or 0, self.service.check_deadline)
                ):
                    raise ReviewRefused("check_execution_admission_expired")
                yield

    def accepted(self, invocation_id: str) -> None:
        # The runtime calls this only after its authority-checked commit_launch.
        with self.service.store._tab.lock:
            self.accepted_id = invocation_id
            self.check.native_execution_receipt_id = invocation_id
            self.check.state = "running"
            self.check.changed.set()

    def settled(self, result) -> None:
        with self.service.store._tab.lock:
            if (
                self.accepted_id is not None
                and result.invocation_id != self.accepted_id
            ):
                raise ValueError("native check receipt mismatch")
            self.native_result = result
            try:
                self.service.register_check_result(
                    self.check.binding.review_id,
                    self.check.proposal_id,
                    result.invocation_id,
                )
            finally:
                if result.cleanup_proven and not self.released:
                    self.release()

    def release(self) -> bool:
        if self.released:
            return True
        if not self.boundary.release():
            return False
        source, report, files = self.reservation._actual()
        try:
            self.reservation.commit(source, report, files)
        except ReviewRefused as exc:
            if exc.reason != "review_expired":
                raise
        finally:
            # Closing the context can invalidate publication, but confirmed native
            # cleanup must still release the retained scratch lease and allowance.
            self.reservation.__exit__(None, None, None)
            self.released = True
        return True


def current_check_execution():
    """Read only at the existing raw-shell dispatch boundary."""
    return _ACTIVE_EXECUTION.get()
