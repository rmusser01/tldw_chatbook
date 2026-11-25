"""A new turn's first call putting the leading system slot back (TASK-42525).

ADR-337 amendment 2026-11-25 (owner-approved), the mirror of the 2026-11-24
removal. When the surface has no rendered-system row (it never had one, or a
fallback's marker removed it) and a new turn's first call starts with a
system prompt, the call is planned as if the slot already led the surface:
every other row meets the existing rules. The slot is written directly after
the tail, before the call's other rows; after its replacements the surface
projection moves the active rendered-system rows first, in sequence order.
This module plans and writes; the service calls it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

#: Routes of a turn's first call, the only calls that may put the slot back.
RESTORING_ROUTES = frozenset({"agent_first", "fresh"})


def _has_slot(entries: tuple[tuple[int, tuple[str, str, str]], ...]) -> bool:
    return any(key[0] == "rendered_system" for _sequence, key in entries)


def restored_slot_projection(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    owner_id: str,
    route_identity: str,
    tail: object | None,
    projection: object,
    descriptors: tuple[object, ...],
    values: tuple[object, ...],
    message_count: int,
) -> object | None:
    """Return the surface as it reads once the slot is back, or None.

    Only the first row qualifies: a request cannot carry a system row after a
    message, and a second one is a changed row the existing rules refuse. The
    slot's artifact is stored now, in the reservation transaction, so the
    planned slot names the bytes the dispatch-binding write will name again.

    Args:
        service: The trace service.
        cursor: Cursor for the caller-owned transaction.
        owner_id: Owner of the segment.
        route_identity: The call's route.
        tail: The surface tail.
        projection: The surface projection at ``tail``.
        descriptors: The request's descriptors, messages first.
        values: The request's values, aligned with ``descriptors``.
        message_count: How many descriptors are messages.

    Returns:
        A projection with the slot first, at the sequence after the tail, when
        the call may put it back; None for any other shape.
    """
    from tldw_chatbook.Chat import console_trace_service as trace

    if (
        route_identity not in RESTORING_ROUTES
        or tail is None
        or not message_count
        or not trace._is_rendered_system_row(descriptors[0], values[0])
        or _has_slot(projection.entries)  # type: ignore[attr-defined]
    ):
        return None
    _kind, reference = service._reference(  # type: ignore[attr-defined]
        cursor, owner_id=owner_id, descriptor=descriptors[0], value=values[0]
    )
    sequence = tail.sequence + 1  # type: ignore[attr-defined]
    return trace._SurfaceProjection(
        tail.node_id,  # type: ignore[attr-defined]
        trace._ProjectionRoot(
            None,
            base=(
                (sequence, ("rendered_system", "artifact", reference.content_id)),
                *projection.entries,  # type: ignore[attr-defined]
            ),
        ),
        lineage_domains=(
            *projection.lineage_domains,  # type: ignore[attr-defined]
            (sequence, "messages_payload"),
        ),
    )


def write_restored_slot(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    owner_id: str,
    segment_id: str,
    tail: object,
    projection: object,
    slot: object,
    value: object,
    route_identity: str,
    reserved_call: object | None,
) -> object:
    """Recheck and write the slot at dispatch binding, before the call's rows.

    Returns:
        The slot node, the surface's new tail.

    Raises:
        ValueError: ``rendered_system_slot_unavailable`` when the call is not
            its turn's first or the surface has a slot already.
    """
    if (
        reserved_call is None
        or route_identity not in RESTORING_ROUTES
        or reserved_call.call_sequence != 0  # type: ignore[attr-defined]
        or _has_slot(projection.entries)  # type: ignore[attr-defined]
    ):
        raise ValueError("rendered_system_slot_unavailable")
    node = service._append_descriptor(  # type: ignore[attr-defined]
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        tail=tail,
        descriptor=slot.descriptor,  # type: ignore[attr-defined]
        value=value,
    )
    service._append_event(  # type: ignore[attr-defined]
        cursor,
        segment_id=segment_id,
        event_type="surface_append",
        surface_node_id=node.node_id,
    )
    return node


def with_slot_first(service: object, root: object, node: object) -> object:
    """Put the written slot first in a checkpoint's projection root."""
    from tldw_chatbook.Chat import console_trace_service as trace

    return trace._ProjectionRoot(
        None,
        base=(
            (node.sequence, service._node_reference_key(node)),  # type: ignore[attr-defined]
            *root.materialize(),  # type: ignore[attr-defined]
        ),
    )


def planned_tail(tail: object) -> object:
    """The tail a slot-first plan builds on: its rows follow the slot."""
    return replace(tail, sequence=tail.sequence + 1)  # type: ignore[attr-defined, type-var]
