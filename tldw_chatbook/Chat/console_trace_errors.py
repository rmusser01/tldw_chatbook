"""Leaf home for the trace-persistence error contract (ADR-097 boot ratchet).

``TraceCallPersistenceError`` was defined in ``console_trace_service``, which
eagerly imports the whole trace stack (final-values, provenance, redaction,
models, repository, semantic revision, transaction observer). Three
boot-resident modules (``console_chat_controller``, ``console_provider_
gateway``, ``console_turn_preparation``) import ONLY this exception from it,
which pulled the entire family onto the boot path. Housing the exception (and
its reservation-status alias) in this dependency-free leaf lets those callers
keep their compile-time import while the trace stack loads on first actual
trace use. ``console_trace_service`` re-exports both names, so existing
``from console_trace_service import TraceCallPersistenceError`` callers keep
working with the SAME class object (identity preserved for except clauses).
"""

from __future__ import annotations

from typing import Literal, TypeAlias

TraceCallReservationStatus: TypeAlias = Literal[
    "not_established", "established", "unknown"
]


class TraceCallPersistenceError(RuntimeError):
    """A content-free pre-dispatch trace write failure."""

    def __init__(
        self,
        *,
        boundary: object | None = None,
        reservation_status: TraceCallReservationStatus | None = None,
    ) -> None:
        super().__init__("Provider call trace persistence failed.")
        self.boundary = boundary
        self.reservation_status = reservation_status
