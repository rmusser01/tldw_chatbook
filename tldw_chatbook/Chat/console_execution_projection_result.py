"""Closed values for the original ConsoleProjection, with exact snapshot reuse.

This is the existing seven-field projection, not an attached-screen model or
authority. Session rails, ancillary display and named parent checkpoints have
their own capture boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .console_execution_protocol import ConsoleProtocolError

if TYPE_CHECKING:
    from .console_control_dispatch import ConsoleProjection


def projection_key(value: ConsoleProjection) -> tuple[int, str, str, int]:
    """Return the original source and transcript invalidation facts."""
    return (
        value.source_revision,
        value.session_id,
        value.incarnation_id,
        value.display_revision,
    )


def encode_projection_payload(
    value: ConsoleProjection, *, messages: list[dict] | None
) -> dict:
    """Encode fresh run/decision values beside full or referenced messages."""
    from .console_control_dispatch import ConsoleProjection
    from .console_execution_decision_values import encode_pending_decision
    from .console_execution_projection_values import encode_run_projection

    if type(value) is not ConsoleProjection:
        raise ConsoleProtocolError("invalid_original_projection")
    return {
        "source_revision": value.source_revision,
        "session_id": value.session_id,
        "incarnation_id": value.incarnation_id,
        "display_revision": value.display_revision,
        "messages": messages,
        "run_state": encode_run_projection(value.run_state),
        "decision": encode_pending_decision(value.decision),
    }


def _shape(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "source_revision",
        "session_id",
        "incarnation_id",
        "display_revision",
        "messages",
        "run_state",
        "decision",
    }:
        raise ConsoleProtocolError("invalid_projection_fields")
    for name in ("source_revision", "display_revision"):
        if type(value[name]) is not int or value[name] < 0:
            raise ConsoleProtocolError("invalid_projection_revision")
    for name in ("session_id", "incarnation_id"):
        if type(value[name]) is not str or not value[name]:
            raise ConsoleProtocolError("invalid_projection_session")
    messages = value["messages"]
    if messages is not None and type(messages) is not list:
        raise ConsoleProtocolError("invalid_projection_messages")


def validate_projection_payload(value: object) -> None:
    """Check finite values without constructing transient transcript messages."""
    from .console_execution_decision_values import decode_pending_decision
    from .console_execution_projection_values import (
        validate_message_projection,
        decode_run_projection,
    )

    _shape(value)
    messages = value["messages"]
    if messages is not None:
        for message in messages:
            validate_message_projection(message)
    decode_run_projection(value["run_state"])
    decision = decode_pending_decision(value["decision"])
    if decision is not None and decision.session_id != value["session_id"]:
        raise ConsoleProtocolError("projection_decision_session_changed")


def decode_projection_payload(
    value: object, *, previous: ConsoleProjection | None = None
) -> ConsoleProjection:
    """Make disposable render values, requiring exact prior snapshot references."""
    from .console_control_dispatch import ConsoleProjection
    from .console_execution_decision_values import decode_pending_decision
    from .console_execution_projection_values import (
        decode_message_projection,
        decode_run_projection,
    )

    _shape(value)
    key = tuple(
        value[name]
        for name in (
            "source_revision",
            "session_id",
            "incarnation_id",
            "display_revision",
        )
    )
    if value["messages"] is None:
        if type(previous) is not ConsoleProjection or projection_key(previous) != key:
            raise ConsoleProtocolError("projection_snapshot_not_owned")
        messages = previous.messages
    else:
        messages = tuple(decode_message_projection(item) for item in value["messages"])
    decision = decode_pending_decision(value["decision"])
    if decision is not None and decision.session_id != value["session_id"]:
        raise ConsoleProtocolError("projection_decision_session_changed")
    return ConsoleProjection(
        *key,
        messages,
        decode_run_projection(value["run_state"]),
        decision,
    )
