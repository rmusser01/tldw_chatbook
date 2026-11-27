"""Trace rules for the calls of Edit & resend, Regenerate and Continue.

ADR-337 amendment 2026-11-26 (owner-approved, TASK-42560). These actions
used to dispatch with Capture Off, and the next captured send in a chat with
project instructions was refused. Each is now traced as a call of its turn:

- the call's request names its action (its route on the direct path; on the
  agent path the run's route descriptor carries it);
- the call re-records its turn: its replacement starts at the turn's user
  row and is proven like a rewritten tail;
- a Continue's request ends with a provider-only continuation instruction
  after the saved reply it extends; the call belongs to the saved user that
  reply answers;
- the next send replaces the action run's own rows after that user row,
  proven by :func:`after_action_call` and the rewritten tail.

This module reads the ledger only; it writes nothing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Chat.console_trace_provenance import (
    TRACED_ACTION_ROUTES,
    ConsoleRequestRoute,
    ProviderArtifactTraceProvenance,
    TraceOmissionReason,
    TraceProvenance,
    TraceProvenanceSource,
    current_turn_source_revision_id,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

#: Durable route identities of a direct-path action call.
ACTION_ROUTE_IDENTITIES = frozenset(route.value for route in TRACED_ACTION_ROUTES)
#: Bound on the calls an action's run may hold after its first call.
_MAX_RUN_CALLS = 256
#: Console's provider-only continuation instruction; the controller's
#: ``CONSOLE_CONTINUE_INSTRUCTION`` (a test pins them equal).
CONTINUE_INSTRUCTION = "Continue and extend the selected message."
#: The actions whose request ends with that instruction: a Continue, and a
#: Regenerate of a continued reply.
_CONTINUING_ACTIONS = frozenset(
    {ConsoleRequestRoute.CONTINUE, ConsoleRequestRoute.REGENERATE}
)


def _saved_row_revision_id(descriptor: TraceProvenance) -> str | None:
    from tldw_chatbook.Chat.console_trace_final_values import (
        rewritten_history_revision_id,
    )

    return rewritten_history_revision_id(descriptor)


def continued_turn_index(
    descriptors: Sequence[TraceProvenance],
    rows: Sequence[Mapping[str, Any]],
    index: int,
    action: ConsoleRequestRoute,
) -> int | None:
    """Return the saved user a continuation instruction at ``index`` answers.

    A Continue (and a Regenerate of a continued reply) ends its rows with
    Console's provider-only instruction after one or more saved assistant
    replies; the call belongs to the saved user turn those replies answer.

    Args:
        descriptors: The request's message descriptors.
        rows: The request's message rows, aligned with ``descriptors``.
        index: The row after which only project context follows.
        action: The action the call names.

    Returns:
        The user's index, or None when the rows have another shape.
    """
    descriptor = descriptors[index]
    content = rows[index].get("content")
    if not (
        action in _CONTINUING_ACTIONS
        and type(descriptor) is ProviderArtifactTraceProvenance
        and descriptor.source is TraceProvenanceSource.ACTIVE_REQUEST
        and rows[index].get("role") == "user"
        # Prompt mix-ins are composed before it.
        and isinstance(content, str)
        and content.endswith(CONTINUE_INSTRUCTION)
    ):
        return None
    position = index - 1
    while (
        position >= 0
        and rows[position].get("role") == "assistant"
        and _saved_row_revision_id(descriptors[position]) is not None
    ):
        position -= 1
    if (
        position == index - 1
        or position < 0
        or rows[position].get("role") != "user"
        or current_turn_source_revision_id(descriptors[position]) is None
    ):
        return None
    return position


def action_turn_revision_id(
    descriptors: Sequence[TraceProvenance], rows: Sequence[Mapping[str, Any]]
) -> str | None:
    """Return the saved user revision a later call of an action's run carries.

    A tool-loop call or resend of a Continue's run carries the saved reply
    after the user, so its last saved row is that reply, not the turn.
    """
    for descriptor, row in zip(reversed(descriptors), reversed(rows), strict=True):
        if row.get("role") == "user" and (
            revision_id := current_turn_source_revision_id(descriptor)
        ):
            return revision_id
    return None


@dataclass(frozen=True, slots=True)
class AfterActionCall:
    """The segment's latest call, as one of an action's run that re-recorded
    its turn (ADR-337 amendment 2026-11-26).

    Args:
        latest_call_id: The segment's latest call.
        origin_call_id: Its run's first call, which re-recorded the turn.
        turn_id: The turn's saved user message.
        pin_revision_id: The user revision the first call pinned.
        user_sequence: Surface sequence of the re-recorded user row; every
            later row was written by the run.
        response_revision_id: The latest call's verified saved response, or
            None when the run ended failed with none the trace can name.
    """

    latest_call_id: str
    origin_call_id: str
    turn_id: str
    pin_revision_id: str
    user_sequence: int
    response_revision_id: str | None


def _boundaries(cursor: sqlite3.Cursor, segment_id: str) -> list[tuple[str, int]]:
    return [
        (str(row[0]), int(row[1]))
        for row in cursor.execute(
            "SELECT call_id, sequence FROM console_trace_events "
            "WHERE segment_id = ? AND event_type = 'call_boundary' "
            "ORDER BY sequence DESC LIMIT ?",
            (segment_id, _MAX_RUN_CALLS + 2),
        ).fetchall()
    ]


def _is_turn_row(cursor: sqlite3.Cursor, node_id: str, pin: str) -> int | None:
    node = cursor.execute(
        "SELECT sequence, component_kind, reference_kind, semantic_revision_id "
        "FROM console_trace_surface_nodes WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    if node is None:
        return None
    sequence, kind, reference, revision_id = node
    if reference == "revision" and revision_id == pin:
        return int(sequence)
    if reference == "artifact" and kind in {"active_request", "prompt_mixins"}:
        return int(sequence)
    return None


def _starts_with_user(
    service: ConsoleTraceService, cursor: sqlite3.Cursor, node_id: str
) -> bool:
    node = cursor.execute(
        "SELECT component_kind, reference_kind, semantic_revision_id, "
        "omission_reason_code FROM console_trace_surface_nodes WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    if node is None:
        return False
    kind, reference, revision_id, omission = node
    if reference == "omission":
        # A chained replacement's final link starts at its placeholders.
        return omission == TraceOmissionReason.REPLACEMENT_CHAIN_LINK.value
    if reference == "artifact":
        return kind in {"active_request", "prompt_mixins"}
    revision = service.repository.get_semantic_revision(cursor, revision_id)
    return revision is not None and revision.normalized_role == "user"


def _never_sent(
    service: ConsoleTraceService, cursor: sqlite3.Cursor, call_id: str
) -> bool:
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    call = service.repository.get_call(cursor, call_id)
    return (
        call is not None
        and call.state is TraceCallState.NOT_DISPATCHED
        and call.surface_node_id is None
        and call.dispatch_started_at is None
    )


def after_action_call(
    service: ConsoleTraceService,
    cursor: sqlite3.Cursor,
    *,
    segment_id: str,
    before_call_id: str | None = None,
) -> AfterActionCall | None:
    """Read the segment's latest call as one of a turn-re-recording run.

    The run's first call wrote its turn's user row as the first row of a
    replacement that started at that turn's earlier user row: an action's
    call does, a normal send never does (its own user row is appended, or
    replaces earlier project or tool rows). Every call after it belongs to
    the run, and the latest call is settled.

    Args:
        service: The trace service.
        cursor: Cursor in the caller-owned transaction.
        segment_id: The segment whose latest call is read.
        before_call_id: At dispatch binding, the reserved call: read the call
            directly before it.

    Returns:
        The evidence, or None for any other shape.
    """
    from tldw_chatbook.Chat.console_trace_continuation import (
        failed_stream_fallback_parent,
        run_chains,
    )
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    boundaries = _boundaries(cursor, segment_id)
    if before_call_id is not None:
        if not boundaries or boundaries[0][0] != before_call_id:
            return None
        boundaries = boundaries[1:]
    while boundaries and _never_sent(service, cursor, boundaries[0][0]):
        # A refused later call of the run, marked not dispatched before it
        # bound any row (ADR-337 amendment 2026-11-26, point 7).
        boundaries = boundaries[1:]
    if not boundaries:
        return None
    latest = service.repository.get_call(cursor, boundaries[0][0])
    if (
        latest is None
        or latest.settled_at is None
        or latest.state
        not in {
            TraceCallState.COMPLETE,
            TraceCallState.STOPPED,
            TraceCallState.ERROR,
            TraceCallState.INTERRUPTED,
        }
    ):
        return None
    origin, runs = run_chains(service.repository, cursor, latest.run_id)
    if origin is None or origin.call_sequence != 0 or origin.route_identity not in {
        "agent_first",
        *ACTION_ROUTE_IDENTITIES,
    }:
        return None
    position = next(
        (index for index, (call_id, _) in enumerate(boundaries) if call_id == origin.call_id),
        None,
    )
    if position is None:
        return None
    pin = cursor.execute(
        "SELECT semantic_revision_id FROM console_trace_events "
        "WHERE call_id = ? AND event_type = 'call_boundary'",
        (origin.call_id,),
    ).fetchall()
    if len(pin) != 1 or pin[0][0] is None:
        return None
    origin_sequence = boundaries[position][1]
    next_sequence = boundaries[position - 1][1] if position else None
    replacements = cursor.execute(
        "SELECT r.start_node_id, r.replacement_node_id "
        "FROM console_trace_events e JOIN console_trace_surface_replacements r "
        "ON r.replacement_id = e.surface_replacement_id "
        "WHERE e.segment_id = ? AND e.event_type = 'surface_replace' "
        "AND e.sequence > ? AND (? IS NULL OR e.sequence < ?) ORDER BY e.sequence",
        (segment_id, origin_sequence, next_sequence, next_sequence),
    ).fetchall()
    user_sequence = next(
        (
            sequence
            for start_id, node_id in replacements
            if (sequence := _is_turn_row(cursor, node_id, pin[0][0]))
            is not None
            and _starts_with_user(service, cursor, start_id)
        ),
        None,
    )
    if user_sequence is None:
        return None
    # Every later call belongs to the run (a llama.cpp resend of one of its
    # calls does too).
    bore = False
    for call_id, _sequence in boundaries[: position + 1]:
        call = service.repository.get_call(cursor, call_id)
        if call is None or (
            call.run_id not in runs
            and getattr(
                failed_stream_fallback_parent(
                    service.repository, cursor, call_id, answered=True
                ),
                "run_id",
                None,
            )
            not in runs
        ):
            return None
        bore |= (
            call.response_started_at is not None
            or service.repository.get_response_link(cursor, call_id) is not None
        )
    link = service.repository.get_response_link(cursor, latest.call_id)
    response = (
        link.semantic_revision_id
        if link is not None
        and link.link_kind == "revision"
        and link.verification_outcome == "verified_equal"
        else None
    )
    if (
        bore
        and response is None
        and (link is None or link.link_kind == "revision")
        and latest.state is not TraceCallState.ERROR
    ):
        # A response the trace cannot name proves nothing. A latest call
        # whose response the trace holds as a provider artifact (a tool call
        # after which the run ended: the emergency stop, a breaker, the turn
        # budget) saved no reply, so the run bore none the next send can
        # carry, as when its call failed.
        return None
    return AfterActionCall(
        latest.call_id,
        origin.call_id,
        origin.turn_id,
        pin[0][0],
        user_sequence,
        response,
    )


def earlier_reply(
    cursor: sqlite3.Cursor,
    *,
    segment_id: str,
    turn_id: str,
    origin_call_id: str,
    revision_id: str,
) -> bool:
    """Whether an earlier call of the turn bore this verified saved reply.

    After a Regenerate whose call failed, the reply it would have replaced
    is back on the active path.
    """
    row = cursor.execute(
        "SELECT 1 FROM console_trace_response_links l "
        "JOIN console_trace_calls c ON c.call_id = l.call_id "
        "JOIN console_trace_events e ON e.call_id = c.call_id "
        "AND e.event_type = 'call_boundary' "
        "JOIN console_trace_events o ON o.call_id = ? "
        "AND o.event_type = 'call_boundary' "
        "WHERE c.segment_id = ? AND c.turn_id = ? AND l.link_kind = 'revision' "
        "AND l.verification_outcome = 'verified_equal' "
        "AND l.semantic_revision_id = ? AND e.sequence < o.sequence LIMIT 1",
        (origin_call_id, segment_id, turn_id, revision_id),
    ).fetchone()
    return row is not None


def unanswered_reply(
    cursor: sqlite3.Cursor, *, conversation_id: str, message_id: str
) -> str | None:
    """Return the parent of a failed reply that holds nothing, else None.

    A Continue or Edit & resend whose call failed leaves its empty failed
    reply on the active path; the next user hangs off it, and the next
    request skips it.
    """
    row = cursor.execute(
        "SELECT parent_message_id FROM messages m WHERE id = ? "
        "AND conversation_id = ? AND deleted = 0 AND role = 'assistant' "
        "AND assistant_generation_state = 'failed' "
        "AND coalesce(content, '') = '' AND image_data IS NULL "
        "AND image_mime_type IS NULL AND provider_continuation_json IS NULL "
        "AND thinking_blocks_json IS NULL AND NOT EXISTS ("
        "SELECT 1 FROM message_attachments a WHERE a.message_id = m.id)",
        (message_id, conversation_id),
    ).fetchone()
    return None if row is None or row[0] is None else str(row[0])


__all__ = [
    "ACTION_ROUTE_IDENTITIES",
    "CONTINUE_INSTRUCTION",
    "AfterActionCall",
    "action_turn_revision_id",
    "after_action_call",
    "continued_turn_index",
    "earlier_reply",
    "unanswered_reply",
]
