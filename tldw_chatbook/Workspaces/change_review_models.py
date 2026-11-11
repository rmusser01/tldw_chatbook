"""Shared Change Review reservation and terminal observation records."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Workspaces.change_turn_tracker import TurnHandle


class ChangeReviewCompletion(Enum):
    """Original terminal disposition; physical idle never implies publication."""

    PENDING = "pending"
    OBSERVED = "observed"
    UNADMITTED = "unadmitted"
    PUBLISHED = "published"
    DISCLOSED_FAILURE = "disclosed_failure"
    CANCELLED = "cancelled"
    UNRESOLVED = "unresolved"


@dataclass
class _Completion:
    disposition: ChangeReviewCompletion = ChangeReviewCompletion.PENDING
    ticket: object = None
    admission_error: str = ""


@dataclass(frozen=True)
class ChangeReviewReservation:
    """Caller-facing handle for one admitted root-ordered window."""

    id: str
    roots: tuple[str, ...]
    _handle: TurnHandle
    _admission_error: str = ""
    _publication_reserved: bool = False
    _terminal_lock: threading.Lock = field(
        default_factory=threading.Lock,
        compare=False,
        repr=False,
    )
    _terminal_claimed: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )

    _completion: _Completion = field(
        default_factory=_Completion, compare=False, repr=False
    )
    _baseline_completion: _Completion = field(
        default_factory=_Completion, compare=False, repr=False
    )
    _on_unadmitted: Callable[[], None] | None = field(
        default=None, compare=False, repr=False
    )
    _notice_sent: threading.Event = field(
        default_factory=threading.Event, compare=False, repr=False
    )

    @property
    def admission_error(self) -> str:
        return self._admission_error or self._completion.admission_error

    def notify_unadmitted(self) -> None:
        """Disclose once to the captured display target, outside native locks."""
        if not self.admission_error:
            return
        with self._terminal_lock:
            if self._notice_sent.is_set() or self._on_unadmitted is None:
                return
            self._notice_sent.set()
        try:
            self._on_unadmitted()
        except Exception:  # noqa: BLE001, S110 - vanished display cannot stop drain
            pass

    @property
    def completion(self) -> ChangeReviewCompletion:
        if (
            self._completion.ticket is not None
            and not self._completion.ticket.retired.is_set()
        ):
            return ChangeReviewCompletion.PENDING
        if self.admission_error:
            return ChangeReviewCompletion.UNADMITTED
        return self._completion.disposition

    @property
    def baseline_completion(self) -> ChangeReviewCompletion:
        """Actual read handoff; the legacy bounded-wait event is not drain proof."""
        if self._admission_error:
            return ChangeReviewCompletion.UNADMITTED
        result = self._baseline_completion
        if result.ticket is not None and not result.ticket.retired.is_set():
            return ChangeReviewCompletion.PENDING
        return result.disposition

    def await_baseline(self, timeout: float = 120.0) -> bool:
        return self._handle.await_baseline(timeout)

    @property
    def baseline_ready(self) -> bool:
        return self._handle._baseline_ready.is_set()

    def _claim_terminal(self) -> bool:
        """Claim the caller-owned terminal path exactly once."""
        with self._terminal_lock:
            if self._terminal_claimed.is_set():
                return False
            self._terminal_claimed.set()
            return True
