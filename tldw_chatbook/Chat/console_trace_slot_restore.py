"""A new turn's first call putting the leading system slot back (TASK-42525).

ADR-337 amendment 2026-11-25 (owner-approved), the mirror of the 2026-11-24
removal. When the surface has no rendered-system row (it never had one, or a
fallback's marker removed it) and a new turn's first call starts with a
system prompt, the call is planned as if the slot already led the surface:
every other row meets the existing rules. The slot is written directly after
the tail, before the call's other rows; after its replacements the surface
projection moves the active rendered-system rows first, in sequence order.
This module plans and writes; the service calls it.

ADR-337 amendment 2026-11-26 (owner-approved, TASK-42550) is the mirror:
when the surface leads with the rendered-system slot and a new turn's first
call has no leading system row (a manual switch from llama.cpp to a hosted
provider), the call is planned as if the slot were gone. Its removal is
written first, with the fallback rewrite's content-free marker; the call's
other rows follow it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import replace

#: Routes of a turn's first call, the only calls that may put the slot back
#: or drop it.
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
    """The tail a slot-change plan builds on: its rows follow the slot or marker."""
    return replace(tail, sequence=tail.sequence + 1)  # type: ignore[attr-defined, type-var]


def _leads_with_slot(projection: object) -> bool:
    entries = projection.entries  # type: ignore[attr-defined]
    return bool(entries) and entries[0][1][:2] == ("rendered_system", "artifact")


def _leads_with_system_row(values: tuple[object, ...]) -> bool:
    return bool(values) and isinstance(values[0], Mapping) and (
        values[0].get("role") == "system"
    )


def dropped_slot_projection(
    *,
    route_identity: str,
    tail: object | None,
    projection: object,
    values: tuple[object, ...],
) -> object | None:
    """Return the surface as it reads once its leading slot is dropped, or None.

    Args:
        route_identity: The call's route.
        tail: The surface tail.
        projection: The surface projection at ``tail``.
        values: The request's message values.

    Returns:
        The projection without its leading rendered-system slot, when a new
        turn's first call has no leading system row; None for any other shape.
    """
    from tldw_chatbook.Chat import console_trace_service as trace

    if (
        route_identity not in RESTORING_ROUTES
        or not _leads_with_slot(projection)
        or _leads_with_system_row(values)
    ):
        return None
    return trace._SurfaceProjection(
        tail.node_id,  # type: ignore[attr-defined]
        trace._ProjectionRoot(None, base=projection.entries[1:]),  # type: ignore[attr-defined]
        lineage_domains=projection.lineage_domains,  # type: ignore[attr-defined]
    )


def drop_planned(service: object, child: object | None, tail: object | None) -> bool:
    """Whether the call was planned with its slot dropped, at binding.

    Such a plan leaves the sequence after the tail to a node written first,
    as a restore's does; the caller has ruled a restore out.
    """
    return (
        child is not None
        and tail is not None
        and service._parent_capabilities[id(child.parent)].next_sequence  # type: ignore[attr-defined]
        == tail.sequence + 2  # type: ignore[attr-defined]
    )


def write_dropped_slot(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    segment_id: str,
    tail: object,
    projection: object,
    bundle: object,
    route_identity: str,
    reserved_call: object | None,
) -> tuple[int, object, object]:
    """Recheck and write the slot's removal at dispatch binding, first.

    Returns:
        The slot's sequence, the removal marker (the surface's new tail) and
        its replacement row.

    Raises:
        ValueError: ``rendered_system_slot_unavailable`` when the call is not
            its turn's first, the slot no longer leads the surface, or the
            request leads with a system row.
    """
    from tldw_chatbook.Chat import console_trace_service as trace
    from tldw_chatbook.Chat.console_trace_fallback_rewrite import remove_surface_slot

    messages = trace._binding(bundle, "messages_payload")
    if (
        reserved_call is None
        or route_identity not in RESTORING_ROUTES
        or reserved_call.call_sequence != 0  # type: ignore[attr-defined]
        or not _leads_with_slot(projection)
        or messages is None
        or _leads_with_system_row(messages.value)
    ):
        raise ValueError("rendered_system_slot_unavailable")
    sequence = projection.entries[0][0]  # type: ignore[attr-defined]
    marker, removal = remove_surface_slot(
        service, cursor, segment_id=segment_id, tail=tail, sequence=sequence
    )
    return sequence, marker, removal
