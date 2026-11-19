"""Data-only Console trace lifecycle writes after the original runtime phase admits them."""

import hmac
import json
from functools import cache

from ..Chat.console_trace_models import TraceCallState, TraceReservationProvenance
from ..Chat.console_trace_redaction import is_stock_writer_sanitizer
from ..Chat.console_trace_repository import ConsoleTraceRepository, TraceCallRecord
from ..Chat.console_trace_settlement import (
    MAX_TRACE_RESPONSE_BYTES, ConsoleTraceSettlementCoordinator, TraceSettlementRequest,
    _PreparedSettlement, _canonical_bytes, _integrity_omission,
)
from .ChaChaNotes_DB import CharactersRAGDB


@cache
def value_types():
    classes = TraceCallState, TraceReservationProvenance, TraceCallRecord, TraceSettlementRequest
    return {cls.__name__: cls for cls in classes}


def stock_source(owner, database):
    return (
        type(owner) is ConsoleTraceSettlementCoordinator and type(owner.repository) is ConsoleTraceRepository
        and not vars(owner).keys() & vars(type(owner)).keys()
        and not vars(owner.repository).keys() & vars(type(owner.repository)).keys()
        and is_stock_writer_sanitizer(owner._sanitizer) and type(database) is CharactersRAGDB
    )


def settlement_request(coordinator, prepared):
    """Detach only a response whose original equality proof authenticates its sanitized bytes."""
    if (
        type(prepared) is not _PreparedSettlement or prepared._control_owner is not None
        or prepared.response_omission is not None
        or prepared.usage_omission is not None
        or prepared.integrity_omission != _integrity_omission(prepared.prior_omission_reason_code, None, None)
        or not is_stock_writer_sanitizer(coordinator._sanitizer)
    ):
        return None
    response = None
    proof = prepared._response_equality_proof
    if prepared.response_bytes is not None:
        if proof is None or not hmac.compare_digest(
            proof, hmac.digest(coordinator._response_equality_key, prepared.response_bytes, "sha256")
        ):
            return None
        response = json.loads(prepared.response_bytes)
    elif proof is not None:
        return None
    # Recheck both payloads now: the original handoff may have been prepared
    # under a different filter before its callback was queued for persistence.
    for value in (response, prepared.usage):
        if value is not None:
            observed = coordinator._sanitizer.sanitize(value)
            if not observed.available or observed.redacted or _canonical_bytes(observed.value) != _canonical_bytes(value):
                return None
    return TraceSettlementRequest(
        prepared.call_id, prepared.outcome, response, prepared.usage, prepared.response_started_at,
        prepared.settled_at, prepared.canonical_message_id, prepared.prior_integrity_state,
        prepared.prior_omission_reason_code,
    )


def execute_trace_operation(function, database, parameters):
    coordinator = ConsoleTraceSettlementCoordinator()
    if function.__name__ == "_write_prepared":
        request = parameters["prepared"]
        if type(request) is not TraceSettlementRequest or len(_canonical_bytes(request.response_envelope)) > MAX_TRACE_RESPONSE_BYTES:
            raise ValueError("Invalid trace settlement.")
        # Repeat original limits and credential sanitization at the storage boundary.
        prepared = coordinator._prepare_safely(request)
        if settlement_request(coordinator, prepared) != request:
            raise ValueError("Trace settlement source changed.")
        parameters = {"prepared": prepared}
    return function(coordinator, database, **parameters)
