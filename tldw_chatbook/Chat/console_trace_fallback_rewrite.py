"""A fallback chain's first call rewriting the run's tail (TASK-42525).

ADR-337 amendment 2026-11-24 (owner-approved). A provider fallback starts a
new chain of its run; its first call carries the run's history for the new
provider. Two differences from the recorded surface are admitted, and only
there:

* the surface's leading rendered-system slot is removed, when the new
  provider takes its system prompt outside the message list;
* the run's own tool rows at the surface's tail are replaced, one for one and
  kind for kind, by the new call's rows: ADR-110's re-encoding for the new
  provider's tool-call format.

Every removed tool row must have been earned by the run (the completed-turn
witness's TASK-42534 rule), at most 256 rows are replaced, and the whole
rewrite is rechecked in the dispatch-binding transaction. This module reads
the ledger only; the service writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import TYPE_CHECKING

from tldw_chatbook.Chat.console_trace_continuation import (
    _MAX_RUN_CHAINS,
    adjacent_call,
    answering_stream_fallback,
    bore_response,
    fallback_parent,
    run_chains,
)
from tldw_chatbook.Chat.console_trace_final_values import (
    FallbackRewriteWitness,
    SurfaceDeltaAdmission,
    VerifiedSurfaceReplacementRange,
)
from tldw_chatbook.Chat.console_trace_models import (
    MAX_SURFACE_REPLACEMENT_SPAN,
    TraceCallState,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    TraceProvenance,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_repository import TraceCallRecord

#: The surface rows a fallback rewrite may replace.
TOOL_ROW_KINDS = frozenset({"tool_call", "tool_result"})


def _refuse() -> ValueError:
    return ValueError("trace_fallback_rewrite_unavailable")


def _is_tool_descriptor(descriptor: TraceProvenance, kind: str) -> bool:
    return (
        type(descriptor) is ProviderArtifactTraceProvenance
        and descriptor.source.value == kind
    )


def earned_tool_rows(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    segment_id: str,
    owner_id: str,
    turn_id: str,
    policy_id: str,
    runs: frozenset[str],
    sequences: tuple[int, ...],
    depth: int = 0,
) -> bool:
    """Whether the run earned every row at ``sequences``.

    A row is earned when a tool-loop call of one of the run's chains appended
    it after a predecessor in its own chain that bore a response (TASK-42534),
    or when a fallback chain's first call wrote it by rewriting rows that are
    themselves earned (ADR-337 amendment 2026-11-24).

    Args:
        service: The trace service.
        cursor: Cursor for the caller-owned transaction.
        segment_id: Segment holding the rows.
        owner_id: The run's owner.
        turn_id: The run's turn.
        policy_id: The run's frozen policy.
        runs: The run's chains (``run_chains``).
        sequences: The rows' sequences, at most 256.
        depth: Rewrites followed so far.

    Returns:
        True only when each row is accounted for exactly once and earned.
    """
    repository = service.repository  # type: ignore[attr-defined]
    if not sequences or depth > _MAX_RUN_CHAINS:
        return False
    rows = cursor.execute(
        """SELECT n.sequence, n.component_kind, n.reference_kind, c.call_id,
                  c.owner_id, c.segment_id, c.turn_id, c.run_id, c.policy_id,
                  c.route_identity, c.call_sequence
             FROM console_trace_surface_nodes n
             LEFT JOIN console_trace_surface_replacements r
                    ON r.replacement_node_id = n.node_id AND r.segment_id = n.segment_id
             JOIN console_trace_events e ON e.segment_id = n.segment_id AND (
                  (e.surface_node_id = n.node_id AND e.event_type = 'surface_append')
                  OR (e.surface_replacement_id = r.replacement_id
                      AND e.event_type = 'surface_replace'))
             JOIN console_trace_calls c ON c.call_id = (
                  SELECT b.call_id FROM console_trace_events b
                   WHERE b.segment_id = e.segment_id AND b.event_type = 'call_boundary'
                     AND b.sequence < e.sequence ORDER BY b.sequence DESC LIMIT 1)
            WHERE n.segment_id = ? AND n.sequence IN ({})
            ORDER BY n.sequence LIMIT 257""".format(",".join("?" for _ in sequences)),
        (segment_id, *sequences),
    ).fetchall()
    if tuple(row[0] for row in rows) != tuple(sorted(sequences)):
        return False
    roots: dict[str, bool] = {}
    for (
        _sequence,
        _component_kind,
        _reference_kind,
        call_id,
        *ownership,
        run_id,
        call_policy_id,
        route_identity,
        call_sequence,
    ) in rows:
        if (
            tuple(ownership) != (owner_id, segment_id, turn_id)
            or call_policy_id != policy_id
            or run_id not in runs
            or route_identity != "tool_loop"
        ):
            return False
        if call_sequence >= 1:
            row_owner, row_segment, row_turn = ownership
            previous = repository.get_call_by_logical_identity(
                cursor,
                owner_id=row_owner,
                segment_id=row_segment,
                turn_id=row_turn,
                run_id=run_id,
                call_sequence=call_sequence - 1,
            )
            if previous is None or (
                not bore_response(previous)
                and answering_stream_fallback(repository, cursor, previous) is None
            ):
                return False
            continue
        if call_id not in roots:
            replaced = _rewritten_range(cursor, segment_id, call_id)
            roots[call_id] = replaced is not None and earned_tool_rows(
                service,
                cursor,
                segment_id=segment_id,
                owner_id=owner_id,
                turn_id=turn_id,
                policy_id=policy_id,
                runs=runs,
                sequences=replaced,
                depth=depth + 1,
            )
        if not roots[call_id]:
            return False
    return True


def rewritten_sequences(
    cursor: sqlite3.Cursor,
    *,
    segment_id: str,
    runs: frozenset[str],
    start: int,
    end: int,
) -> set[int]:
    """Sequences in ``[start, end]`` that a run's fallback rewrite replaced.

    Args:
        cursor: Cursor for the caller-owned transaction.
        segment_id: Segment holding the rows.
        runs: The run's chains.
        start: First sequence of the range.
        end: Last sequence of the range.

    Returns:
        Every sequence a tool-row replacement by a fallback chain's first call
        of one of ``runs`` covers inside the range.
    """
    from tldw_chatbook.Chat.console_trace_service import SURFACE_REMOVAL_COMPONENT

    rows = cursor.execute(
        """SELECT r.start_sequence, r.end_sequence, c.run_id
             FROM console_trace_surface_replacements r
             JOIN console_trace_surface_nodes n ON n.node_id = r.replacement_node_id
             JOIN console_trace_events e ON e.segment_id = r.segment_id
                  AND e.surface_replacement_id = r.replacement_id
                  AND e.event_type = 'surface_replace'
             JOIN console_trace_calls c ON c.call_id = (
                  SELECT b.call_id FROM console_trace_events b
                   WHERE b.segment_id = e.segment_id AND b.event_type = 'call_boundary'
                     AND b.sequence < e.sequence ORDER BY b.sequence DESC LIMIT 1)
            WHERE r.segment_id = ? AND n.component_kind != ?
              AND c.route_identity = 'tool_loop' AND c.call_sequence = 0
              AND r.end_sequence >= ? AND r.start_sequence <= ?
            LIMIT 257""",
        (segment_id, SURFACE_REMOVAL_COMPONENT, start, end),
    ).fetchall()
    return {
        sequence
        for low, high, run_id in rows
        if run_id in runs
        for sequence in range(max(low, start), min(high, end) + 1)
    }


def _rewritten_range(
    cursor: sqlite3.Cursor, segment_id: str, call_id: str
) -> tuple[int, ...] | None:
    """Return the tool rows one fallback chain's first call replaced, if any."""
    from tldw_chatbook.Chat.console_trace_service import SURFACE_REMOVAL_COMPONENT

    boundary = cursor.execute(
        "SELECT sequence FROM console_trace_events WHERE segment_id = ? "
        "AND event_type = 'call_boundary' AND call_id = ?",
        (segment_id, call_id),
    ).fetchall()
    if len(boundary) != 1:
        return None
    rows = cursor.execute(
        """SELECT r.start_sequence, r.end_sequence
             FROM console_trace_events e
             JOIN console_trace_surface_replacements r
               ON r.replacement_id = e.surface_replacement_id
             JOIN console_trace_surface_nodes n ON n.node_id = r.replacement_node_id
            WHERE e.segment_id = ? AND e.event_type = 'surface_replace'
              AND e.sequence > ? AND n.component_kind != ?
              AND e.sequence < COALESCE((
                  SELECT MIN(b.sequence) FROM console_trace_events b
                   WHERE b.segment_id = ? AND b.event_type = 'call_boundary'
                     AND b.sequence > ?), 9223372036854775807)
            LIMIT 2""",
        (
            segment_id,
            boundary[0][0],
            SURFACE_REMOVAL_COMPONENT,
            segment_id,
            boundary[0][0],
        ),
    ).fetchall()
    if len(rows) != 1:
        return None
    start, end = rows[0]
    return tuple(range(start, end + 1))


def remove_surface_slot(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    segment_id: str,
    tail: object,
    sequence: int,
) -> tuple[object, object]:
    """Remove one surface row with a content-free marker (ADR-337, 2026-11-24).

    The replacement points at a removal marker, which every projection drops,
    so nothing takes the row's place.

    Returns:
        The marker node, now the segment's tail, and the replacement row.
    """
    from tldw_chatbook.Chat.console_trace_models import (
        SurfaceReplacement,
        TraceOmission,
    )
    from tldw_chatbook.Chat.console_trace_provenance import TraceOmissionReason
    from tldw_chatbook.Chat.console_trace_service import SURFACE_REMOVAL_COMPONENT

    repository = service.repository  # type: ignore[attr-defined]
    (slot,) = repository.read_lineage_surface_nodes(
        cursor, segment_id=segment_id, start_sequence=sequence, end_sequence=sequence
    )
    marker = repository.append_surface_node(
        cursor,
        segment_id=segment_id,
        sequence=tail.sequence + 1,  # type: ignore[attr-defined]
        predecessor_node_id=tail.node_id,  # type: ignore[attr-defined]
        component_kind=SURFACE_REMOVAL_COMPONENT,
        reference=TraceOmission(
            SURFACE_REMOVAL_COMPONENT, TraceOmissionReason.SURFACE_SLOT_REMOVED.value
        ),
    )
    stored = repository.append_surface_replacement(
        cursor,
        segment_id=segment_id,
        replacement=SurfaceReplacement(
            predecessor_head_id=tail.node_id,  # type: ignore[attr-defined]
            start_node_id=slot.node_id,
            end_node_id=slot.node_id,
            start_sequence=sequence,
            end_sequence=sequence,
            replacement_node_id=marker.node_id,
        ),
    )
    service._append_event(  # type: ignore[attr-defined]
        cursor,
        segment_id=segment_id,
        event_type="surface_replace",
        surface_replacement_id=stored.replacement_id,
    )
    return marker, stored


def validate_fallback_rewrite(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    witness: FallbackRewriteWitness,
    owner_id: str,
    segment_id: str,
    plan: object | None,
    descriptors: tuple[TraceProvenance, ...],
    reserved_call: TraceCallRecord | None = None,
) -> None:
    """Recheck one fallback rewrite against the ledger (ADR-337, 2026-11-24).

    Args:
        service: The trace service.
        cursor: Cursor for the caller-owned transaction, before any write.
        witness: The rewrite.
        owner_id: The owner of the segment.
        segment_id: The segment being rewritten.
        plan: The tool rows' replacement range, or None (slot removal only).
        descriptors: The new tool rows.
        reserved_call: The call being bound; None while planning, where the
            caller has just admitted it as a fallback chain's first call.

    Raises:
        ValueError: ``trace_fallback_rewrite_unavailable`` on any mismatch.
    """
    repository = service.repository  # type: ignore[attr-defined]
    failed = repository.get_call(cursor, witness.failed_call_id)
    if failed is None or (failed.owner_id, failed.segment_id) != (
        owner_id,
        segment_id,
    ):
        raise _refuse()
    if reserved_call is not None:
        # Only a fallback chain's first call may rewrite: sequence 0, directly
        # after the failed call of another chain of its actor.
        if (
            reserved_call.call_sequence != 0
            or reserved_call.state is not TraceCallState.RESERVED
            or fallback_parent(
                repository,
                cursor,
                adjacent_call(repository, cursor, reserved_call, later=False),
                run_id=reserved_call.run_id,
            )
            != failed
            or (
                reserved_call.owner_id,
                reserved_call.segment_id,
                reserved_call.turn_id,
                reserved_call.policy_id,
            )
            != (failed.owner_id, failed.segment_id, failed.turn_id, failed.policy_id)
        ):
            raise _refuse()
    _origin, runs = run_chains(repository, cursor, failed.run_id)
    tail = service._effective_surface_tail(cursor, segment_id)  # type: ignore[attr-defined]
    if not runs or tail is None or failed.surface_node_id != tail.node_id:
        raise _refuse()
    entries = service._surface_projection(cursor, segment_id, tail).entries  # type: ignore[attr-defined]
    if witness.removed_slot_sequence is not None:
        # The slot is the run's own: its failed call's header carried it.
        header = repository.get_request_header(cursor, failed.request_header_id)
        if (
            not entries
            or entries[0][0] != witness.removed_slot_sequence
            or entries[0][1][:2] != ("rendered_system", "artifact")
            or header is None
            or not any(
                item.component_kind == "rendered_system_row"
                for item in header.components
            )
        ):
            raise _refuse()
    if plan is None:
        if descriptors or witness.descriptors:
            raise _refuse()
        return
    start = int(plan.start_sequence)  # type: ignore[attr-defined]
    end = int(plan.end_sequence)  # type: ignore[attr-defined]
    removed = tuple(entry for entry in entries if start <= entry[0] <= end)
    if (
        getattr(plan, "chain", ())
        or getattr(plan, "predecessor_head_id", None) != tail.node_id
        or end - start + 1 > MAX_SURFACE_REPLACEMENT_SPAN
        or tuple(sequence for sequence, _ in removed) != tuple(range(start, end + 1))
        or tuple(descriptors) != witness.descriptors
        or any(
            key[0] not in TOOL_ROW_KINDS
            or key[1] != "artifact"
            or not _is_tool_descriptor(descriptor, key[0])
            for (_sequence, key), descriptor in zip(removed, descriptors, strict=True)
        )
        or not earned_tool_rows(
            service,
            cursor,
            segment_id=segment_id,
            owner_id=owner_id,
            turn_id=failed.turn_id,
            policy_id=failed.policy_id,
            runs=runs,
            sequences=tuple(range(start, end + 1)),
        )
    ):
        raise _refuse()


def prepare_fallback_rewrite(
    service: object,
    cursor: sqlite3.Cursor,
    *,
    failed: TraceCallRecord,
    owner_id: str,
    segment_id: str,
    route_identity: str,
    preparation_identity: str,
    provenance: object,
    values: tuple[object, ...],
    tail: object,
    projection: object,
    active: tuple[tuple[int, int, tuple[str, str, str]], ...],
    matches: object,
    retained_artifact_values: dict[int, object],
    rendered_system: object,
    current_policy_id: str | None,
    reserved_call: TraceCallRecord | None,
    known_credentials: tuple[str, ...],
) -> tuple[SurfaceDeltaAdmission, object]:
    """Plan a fallback chain's first call whose request rewrites the tail.

    Args:
        service: The trace service.
        cursor: Cursor for the caller-owned transaction.
        failed: The failed call the new chain follows.
        owner_id: Owner of the segment.
        segment_id: Segment holding the run's surface.
        route_identity: The call's route; must be ``tool_loop``.
        preparation_identity: Identity of the prepared request.
        provenance: The request's provenance.
        values: The request's provider values.
        tail: The surface tail.
        projection: The surface projection at ``tail``.
        active: Active entries as (physical ordinal, sequence, key), in
            provider-domain order.
        matches: The planner's exact row comparison.
        retained_artifact_values: The planner's retained artifact values.
        rendered_system: Returns the leading slot the planner matched, if any.
        current_policy_id: The call's frozen policy.
        reserved_call: An owned reservation being rebuilt, if any.
        known_credentials: Transient secrets for artifact comparison.

    Returns:
        The admission and its verification boundary.

    Raises:
        ValueError: ``trace_fallback_rewrite_unavailable`` for any other shape.
    """
    from tldw_chatbook.Chat import console_trace_service as trace

    messages = tuple(provenance.messages_payload)  # type: ignore[attr-defined]
    descriptors = messages + tuple(provenance.continuations)  # type: ignore[attr-defined]
    if (
        route_identity != "tool_loop"
        or tail is None
        or current_policy_id is None
        or len(descriptors) != len(messages)
    ):
        raise _refuse()
    removed_slot = None
    offset = 0
    if (
        active
        and active[0][0] == 0
        and active[0][2][:2] == ("rendered_system", "artifact")
        and not (
            messages
            and (
                trace._is_rendered_system_row(messages[0], values[0])
                or getattr(values[0], "get", lambda _key: None)("role") == "system"
            )
        )
    ):
        removed_slot = active[0][1]
        offset = 1
    rest = active[offset:]
    prefix = 0
    while prefix < min(len(rest), len(descriptors)) and matches(prefix + offset, prefix):  # type: ignore[operator]
        prefix += 1
    changed = rest[prefix:]
    new_rows = descriptors[prefix:]
    if len(changed) != len(new_rows) or (removed_slot is None and not changed):
        raise _refuse()
    entries = tuple(
        entry
        for entry in projection.entries  # type: ignore[attr-defined]
        if entry[0] != removed_slot
    )
    replacement_range = None
    if changed:
        start, end = changed[0][1], changed[-1][1]
        nodes = {
            node.sequence: node
            for node in service.repository.read_lineage_surface_nodes(  # type: ignore[attr-defined]
                cursor, segment_id=segment_id, start_sequence=start, end_sequence=end
            )
        }
        position = next(
            (index for index, entry in enumerate(entries) if entry[0] == start), None
        )
        if start not in nodes or end not in nodes or position is None:
            raise _refuse()
        replacement_range = VerifiedSurfaceReplacementRange(
            predecessor_head_id=tail.node_id,  # type: ignore[attr-defined]
            start_node_id=nodes[start].node_id,
            end_node_id=nodes[end].node_id,
            start_sequence=start,
            end_sequence=end,
            current_ordinal=position,
            component_name="messages_payload",
            component_ordinal=sum(
                trace._surface_reference_domain(key) == "messages_payload"
                for _sequence, key in entries[:position]
            ),
        )
    witness = FallbackRewriteWitness(failed.call_id, removed_slot, tuple(new_rows))
    validate_fallback_rewrite(
        service,
        cursor,
        witness=witness,
        owner_id=owner_id,
        segment_id=segment_id,
        plan=replacement_range,
        descriptors=witness.descriptors,
    )
    # A cold parent built from committed references, without the removed slot.
    checkpoint = service._completed_tool_turn_parent(  # type: ignore[attr-defined]
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        tail=tail,
        projection=trace._SurfaceProjection(
            tail.node_id,  # type: ignore[attr-defined]
            trace._ProjectionRoot(None, base=entries),
            lineage_domains=projection.lineage_domains,  # type: ignore[attr-defined]
        ),
        route=route_identity,
        policy_id=current_policy_id,
        retained_descriptors={
            sequence: descriptors[index]
            for index, (_ordinal, sequence, _key) in enumerate(rest[:prefix])
        },
    )
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=tail.node_id,  # type: ignore[attr-defined]
        route_identity=route_identity,
        preparation_identity=preparation_identity,
        descriptors=witness.descriptors,
        projection_checkpoint=checkpoint,
        replacement_range=replacement_range,
        fallback_rewrite=witness,
    )
    tool_loop = tuple(provenance.tool_loop)  # type: ignore[attr-defined]
    delta_provenance = replace(
        provenance,
        messages=(
            new_rows
            if provenance.messages == provenance.messages_payload  # type: ignore[attr-defined]
            else provenance.messages  # type: ignore[attr-defined]
        ),
        messages_payload=new_rows,
        continuations=(),
        tool_loop=tuple(
            position
            for position, index in enumerate(range(prefix, len(descriptors)))
            if index in tool_loop
        ),
    )
    boundary = service.prepare_surface_provenance(  # type: ignore[attr-defined]
        cursor,
        checkpoint,
        provenance=delta_provenance,
        admission=admission,
        values=tuple(values[prefix:]),
        reserved_call=reserved_call,
        known_credentials=known_credentials,
        retained_artifact_values=retained_artifact_values,
        rendered_system=None if removed_slot is not None else rendered_system(),  # type: ignore[operator]
    )
    return admission, boundary
