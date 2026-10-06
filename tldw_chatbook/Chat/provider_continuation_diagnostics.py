"""Content-free reason codes for provider-continuation persistence failures.

TASK-34100.3 (gap-02). The agent loop persists a provider continuation
through ``ConsoleChatStore.persist_provider_continuation_event``. Each of the
store's preconditions raises a ``RuntimeError`` with a fixed, code-owned
sentence, but the loop used to swallow that exception, so a live failure
could not say which precondition failed.

ADR-029 treats an exception's message as private (only its class name is
metadata), so the loop never logs the text itself. Instead it maps the fixed
sentence to one of the codes below and logs the code. Text that is not one of
these sentences maps to ``UNRECOGNIZED_STORE_FAILURE``; nothing a caller put in
an exception can reach the log through this module.

``Tests/Agents/test_continuation_error_reasons.py`` reads the store's source
and fails when a new failure sentence on the persist path has no code here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

UNRECOGNIZED_STORE_FAILURE = "unrecognized"

STORE_FAILURE_CODES: Mapping[str, str] = MappingProxyType(
    {
        # persist_provider_continuation_event: ownership and storage.
        "Durable continuation needs a distinct assistant owner.": "owner_not_primary",
        "Durable continuation owner is unavailable.": "owner_unavailable",
        "Provider continuation could not be saved; retry or discard the "
        "interrupted run.": "storage_unavailable",
        "Durable managed continuation owner is unavailable.": "managed_owner_unavailable",
        "Durable continuation state is unavailable.": "state_unserializable",
        "Durable continuation storage is unavailable.": "create_unavailable",
        "Continuation version conflict; reload and retry.": "version_conflict",
        "Durable continuation version is unavailable.": "version_unavailable",
        # persist_provider_continuation_event: dispatch-recovery handoff.
        "Provider continuation handoff is unavailable.": "handoff_unavailable",
        "Provider continuation handoff failed.": "handoff_failed",
        "Provider continuation handoff conflicted; reload and retry.": "handoff_conflict",
        "Provider continuation terminal settlement failed.": (
            "terminal_settlement_failed"
        ),
        # _continuation_event_value.
        "Continuation revision conflict.": "revision_conflict",
        "Unsupported continuation event.": "unsupported_event",
        # _settle_dispatch_recovery.
        "Dispatch recovery owner changed during settlement.": (
            "settlement_owner_changed"
        ),
        # ensure_provider_continuation_durable (raised as durability.reason).
        "Local continuation storage is unavailable; save the message and retry.": (
            "local_storage_unavailable"
        ),
        "Local continuation intent is stale or unavailable; save and retry.": (
            "local_intent_stale"
        ),
        "Portable sync projection is unavailable; restore sync configuration.": (
            "sync_projection_unavailable"
        ),
        "Portable sync needs a file-backed state repository; configure one and "
        "retry.": "sync_repository_not_durable",
        "Portable sync source does not match local continuation storage.": (
            "sync_source_mismatch"
        ),
        "Portable sync projection failed; restore local sync state and retry.": (
            "sync_projection_failed"
        ),
        "Portable sync receipt is missing or inconsistent; reconcile and retry.": (
            "sync_receipt_inconsistent"
        ),
    }
)


#: The runtime could not project the checkpoint, so the store was never called.
PROJECTION_FAILED = "projection_failed"

#: Failures raised before the store wrote the continuation or settled the
#: dispatch (a failed terminal settlement is one atomic, rolled-back write).
#: The durability-barrier codes and ``settlement_owner_changed`` are absent on
#: purpose: they are raised after a commit. ``UNRECOGNIZED_STORE_FAILURE`` is
#: absent too, so anything unknown keeps failing closed.
UNWRITTEN_STORE_FAILURES = frozenset(
    {
        PROJECTION_FAILED,
        "owner_not_primary",
        "owner_unavailable",
        "storage_unavailable",
        "managed_owner_unavailable",
        "state_unserializable",
        "create_unavailable",
        "version_conflict",
        "version_unavailable",
        "terminal_settlement_failed",
        "revision_conflict",
        "unsupported_event",
    }
)


#: The two ``field_states`` key prefixes a continuation failure is traced
#: under. Trace ``field_states`` VALUES are the per-field state vocabulary
#: (observed, redacted, truncated, omitted, not_available, capture_failed)
#: that Trace export and import enforce, so a code stored as a value made the
#: run's conversation unexportable. The code rides in the key instead, with
#: the legal state ``observed``: ``{"continuation_error:batch_persist_failed":
#: "observed"}``.
CONTINUATION_ERROR_FIELD = "continuation_error"
CONTINUATION_STORE_FAILURE_FIELD = "continuation_store_failure"
_REASON_STATE = "observed"


def continuation_reason_states(
    reason: str, store_failure: str | None = None
) -> dict[str, str]:
    """Return the trace ``field_states`` entries that name a failure.

    Args:
        reason: The runtime's code for the continuation exit that was taken.
        store_failure: The store's precondition code, when a persist failed.

    Returns:
        One entry per code, keyed ``<field>:<code>`` with a legal Trace state.
    """
    states = {f"{CONTINUATION_ERROR_FIELD}:{reason}": _REASON_STATE}
    if store_failure is not None:
        states[f"{CONTINUATION_STORE_FAILURE_FIELD}:{store_failure}"] = _REASON_STATE
    return states


def continuation_reasons(field_states: Mapping[str, object]) -> dict[str, str]:
    """Read back the codes ``continuation_reason_states`` recorded.

    Args:
        field_states: A trace step's or record's ``field_states``.

    Returns:
        ``{"continuation_error": code}`` plus ``continuation_store_failure``
        when present; empty when the step names no continuation failure.
    """
    reasons: dict[str, str] = {}
    for key, state in field_states.items():
        field, separator, code = str(key).partition(":")
        if (
            separator
            and code
            and state == _REASON_STATE
            and field in {CONTINUATION_ERROR_FIELD, CONTINUATION_STORE_FAILURE_FIELD}
        ):
            reasons[field] = code
    return reasons


def store_failure_code(exc: BaseException) -> str:
    """Name the store precondition behind ``exc`` without exposing its text.

    Args:
        exc: The exception raised while persisting a continuation event.

    Returns:
        The code for one of the store's fixed failure sentences, or
        ``UNRECOGNIZED_STORE_FAILURE`` for anything else.
    """
    if not isinstance(exc, RuntimeError) or len(exc.args) != 1:
        return UNRECOGNIZED_STORE_FAILURE
    message = exc.args[0]
    if type(message) is not str:
        return UNRECOGNIZED_STORE_FAILURE
    return STORE_FAILURE_CODES.get(message, UNRECOGNIZED_STORE_FAILURE)
