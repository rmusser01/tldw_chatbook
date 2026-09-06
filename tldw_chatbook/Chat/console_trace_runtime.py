"""Production ownership for normalized Console provider-call boundaries."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone

from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.console_trace_final_values import CompletedToolTurnWitness
from tldw_chatbook.Chat.console_trace_models import TraceCallState, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    DerivedTraceProvenance,
    ProviderRequestProvenance,
    RequestRouteTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenance,
    frozen_policy_from_provenance,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    TraceCallRecord,
)
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceCallBoundary,
    ConsoleTraceService,
    TraceCallIdentity,
    TraceCallPersistenceError,
)
from tldw_chatbook.DB.base_db import operation_owned_connection

REVISION_OWNER_LOOKUP_BATCH_SIZE = 256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _saved_revision_ids(descriptor: TraceProvenance) -> Iterator[str]:
    if type(descriptor) is SavedRevisionTraceProvenance:
        yield descriptor.revision_id
    elif type(descriptor) is DerivedTraceProvenance:
        for nested in descriptor.inputs:
            yield from _saved_revision_ids(nested)


class ConsoleTraceBoundaryFactory:
    """Create normalized call boundaries from provider-prepared requests.

    One process-wide instance owns the service surface capability cache. The
    database remains authoritative; the cache only avoids replaying unchanged
    provider values when extending a live segment.

    Args:
        database: Transaction-owning Chat database.
        repository: Optional shared normalized trace repository.
        service: Optional shared trace service.
    """

    def __init__(
        self,
        database: object,
        *,
        repository: ConsoleTraceRepository | None = None,
        service: ConsoleTraceService | None = None,
    ) -> None:
        self.database = database
        self.repository = repository or ConsoleTraceRepository()
        self.service = service or ConsoleTraceService(self.repository)
        self._lock = threading.RLock()

    def _verify_owned_reservation(
        self,
        cursor: sqlite3.Cursor,
        boundary: ConsoleTraceCallBoundary,
        accepted_preparation: object,
    ) -> TraceCallRecord:
        """Read exact live ownership without issuing or retiring a capability."""
        if (
            type(boundary) is not ConsoleTraceCallBoundary
            or boundary._factory is not self
            or boundary._accepted_preparation is not accepted_preparation
            or accepted_preparation is None
            or boundary.dispatch_started
            or boundary._recovery_transferred
            or boundary.dispatch_outcome == "unknown"
            or boundary._reserved is None
            or not isinstance(boundary._request, PreparedProviderRequest)
            or boundary._request.semantic.provenance is None
        ):
            raise ValueError("trace_recovery_owner")
        reserved = self.repository.get_call(cursor, boundary._reserved.call_id)
        latest = self.repository.get_latest_call_boundary(cursor, boundary.identity.segment_id)
        tail = self.service._effective_surface_tail(cursor, boundary.identity.segment_id)
        if (
            reserved != boundary._reserved
            or reserved is None
            or reserved.state is not TraceCallState.RESERVED
            or latest is None or latest.call_id != reserved.call_id
            or (None if tail is None else tail.node_id)
            != boundary.admission.predecessor_surface_head_id
            or any(value is not None for value in (
                reserved.surface_node_id, reserved.request_header_id,
                reserved.provider_name, reserved.model_name, reserved.route_identity,
                reserved.dispatch_started_at, reserved.response_started_at,
                reserved.settled_at, reserved.provider_inactive_at,
                reserved.outcome, reserved.usage,
            ))
            or self.repository.get_response_link(cursor, reserved.call_id) is not None
            or cursor.execute(
                "SELECT COUNT(*) FROM console_trace_events WHERE event_type = 'call_boundary' AND call_id = ?",
                (reserved.call_id,),
            ).fetchone()[0] != 1
        ):
            raise ValueError("trace_recovery_reservation")
        revision = self.repository.get_semantic_revision(cursor, boundary._current_revision_id)
        owner = self.repository.get_owner(cursor, reserved.owner_id)
        policy = frozen_policy_from_provenance(boundary._request.semantic.provenance)
        if (
            revision is None or owner is None
            or revision.source_message_id != reserved.turn_id
            or self.repository.get_attached_owner_by_conversation(
                cursor, revision.source_conversation_id,
            ) != owner
            or policy.policy_id != reserved.policy_id
            or self.repository.get_policy(cursor, policy.policy_id) != policy
        ):
            raise ValueError("trace_recovery_revision")
        return reserved

    def _verify_owned_recovery(
        self, boundary: ConsoleTraceCallBoundary, accepted_preparation: object,
    ) -> None:
        """Prove safe local re-entry; transport still requires full preparation."""
        try:
            with self._lock, operation_owned_connection(self.database), self.database.transaction() as cursor:
                self._verify_owned_reservation(cursor, boundary, accepted_preparation)
        except Exception:  # noqa: BLE001 - failed proof cannot expose database/provider details
            raise TraceCallPersistenceError(boundary=boundary) from None

    def _recover_owned_boundary(
        self,
        boundary: ConsoleTraceCallBoundary,
        accepted_preparation: object,
        request: PreparedProviderRequest,
        resolution: object,
        route: object,
    ) -> ConsoleTraceCallBoundary:
        """Reverify the exact live owner's reservation without allocating a call."""
        try:
            if (
                type(boundary) is not ConsoleTraceCallBoundary
                or boundary._factory is not self
                or boundary._accepted_preparation is not accepted_preparation
                or accepted_preparation is None
                or boundary._request != request
                or boundary._resolution != resolution
                or boundary.admission.route_identity != getattr(route, "value", None)
                or boundary.dispatch_started
                or boundary._recovery_transferred
                or boundary.dispatch_outcome == "unknown"
                or boundary._reserved is None
                or request.provenance is None
                or request.semantic.provenance is None
            ):
                raise ValueError("trace_recovery_owner")
            with self._lock, operation_owned_connection(self.database):
                with self.database.transaction(immediate=True) as cursor:
                    reserved = self._verify_owned_reservation(
                        cursor, boundary, accepted_preparation,
                    )
                    # Remove the original verifier before a replacement can exist.
                    boundary._retired = True
                    self.service._retire_preparation(boundary.admission)
                    admission, surface = self.service.prepare_current_surface_delta(
                        cursor,
                        owner_id=reserved.owner_id, segment_id=reserved.segment_id,
                        route_identity=boundary.admission.route_identity,
                        preparation_identity=new_opaque_id(), provenance=request.provenance,
                        values=tuple(request.messages_payload) + tuple(
                            group.checkpoint for group in request.continuation_groups
                        ),
                        completed_tool_turn=boundary.admission.completed_tool_turn,
                        current_turn_id=reserved.turn_id, current_policy_id=reserved.policy_id,
                        reserved_call=reserved,
                    )
                recovered = ConsoleTraceCallBoundary(
                    service=self.service, database=self.database, identity=boundary.identity,
                    admission=admission, occurred_at_factory=_utc_now,
                    surface_boundary=surface, _reserved=reserved,
                    _factory=self, _request=request, _resolution=resolution,
                    _current_revision_id=boundary._current_revision_id,
                    _accepted_preparation=accepted_preparation,
                )
                # Verifier retirement happens before reconstruction, but the
                # reservation owner transfers only once a replacement exists.
                boundary._recovery_transferred = True
                return recovered
        except Exception:  # noqa: BLE001 - recovery errors are content-free owned failures
            raise TraceCallPersistenceError(boundary=boundary) from None

    def __call__(
        self,
        request: PreparedProviderRequest,
        _resolution: object,
        route: object,
    ) -> ConsoleTraceCallBoundary:
        """Create one durable boundary for a prepared provider request.

        Args:
            request: Prepared request carrying normalized trace provenance.
            _resolution: Reserved provider resolution argument.
            route: Dispatch route, when the caller has resolved one.

        Returns:
            ConsoleTraceCallBoundary: Boundary owned by the request's current turn.

        Raises:
            TypeError: If ``request`` is not a prepared provider request.
            ValueError: If required trace provenance or ownership is unavailable.
        """
        if not isinstance(request, PreparedProviderRequest):
            raise TypeError("request")
        provenance = request.provenance
        semantic_provenance = request.semantic.provenance
        if not isinstance(provenance, ProviderRequestProvenance):
            raise ValueError("trace_provenance_unavailable")
        if semantic_provenance is None:
            raise ValueError("trace_policy_unavailable")
        route_record = next(
            (
                item
                for item in provenance.metadata
                if type(item) is RequestRouteTraceProvenance
            ),
            None,
        )
        if route_record is None:
            raise ValueError("trace_route_unavailable")
        route_identity = route_record.route.value
        requested_route = getattr(route, "value", None)
        if requested_route is not None and requested_route != route_identity:
            if route is not ConsoleRequestRoute.LLAMA_FALLBACK:
                raise ValueError("trace_route_mismatch")
            fallback_route = request_route_provenance(route)
            provenance = replace(
                provenance,
                metadata=tuple(
                    fallback_route
                    if type(item) is RequestRouteTraceProvenance
                    else item
                    for item in provenance.metadata
                ),
            )
            route_record = fallback_route
            route_identity = fallback_route.route.value
        message_revision_ids = tuple(
            revision_id
            for descriptor in provenance.messages_payload
            for revision_id in _saved_revision_ids(descriptor)
        )
        if not message_revision_ids:
            raise ValueError("trace_owner_unavailable")
        active_descriptor_index = -1
        if route_record.route is ConsoleRequestRoute.DIRECT_PREFILL:
            if (
                len(provenance.messages_payload) < 2
                or not request.messages_payload
                or request.messages_payload[-1].get("role") != "assistant"
            ):
                raise ValueError("trace_prefill_unavailable")
            # The final provider row is an unsaved response prefill. The
            # preceding active user revision still owns the durable turn.
            active_descriptor_index = -2
        active_revision_ids = tuple(
            _saved_revision_ids(provenance.messages_payload[active_descriptor_index])
        )
        tool_loop = route_record.route is ConsoleRequestRoute.TOOL_LOOP
        if tool_loop:
            # Tool results are provider artifacts, not a new saved user turn.
            # This candidate must match the durable chain origin below; it is
            # never sufficient on its own to admit continuation ownership.
            active_revision_ids = message_revision_ids[-1:]
        if len(active_revision_ids) != 1:
            raise ValueError("trace_turn_unavailable")
        revision_ids = message_revision_ids + tuple(
            revision_id
            for descriptor in provenance.continuations
            for revision_id in _saved_revision_ids(descriptor)
        )
        policy = frozen_policy_from_provenance(semantic_provenance)
        preparation_identity = new_opaque_id()
        # Both route identities are canonical UUIDv4 strings. Bind their pair
        # durably, without a process cache or a second ownership table.
        run_id = (
            f"{route_record.actor_id}:{route_record.chain_id}"
            if route_record.chain_id is not None
            else new_opaque_id()
        )
        idempotency_key = new_opaque_id()
        call_sequence = 0
        reserved = None
        with self._lock, operation_owned_connection(self.database):
            with self.database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                unique_revision_ids = tuple(dict.fromkeys(revision_ids))
                rows = []
                for offset in range(
                    0,
                    len(unique_revision_ids),
                    REVISION_OWNER_LOOKUP_BATCH_SIZE,
                ):
                    batch = unique_revision_ids[
                        offset : offset + REVISION_OWNER_LOOKUP_BATCH_SIZE
                    ]
                    rows.extend(
                        cursor.execute(
                            """SELECT revision_id, source_conversation_id, source_message_id
                                 FROM console_trace_semantic_revisions
                                WHERE revision_id IN ({})""".format(
                                ",".join("?" for _ in batch)
                            ),
                            batch,
                        ).fetchall()
                    )
                by_revision = {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
                if any(revision_id not in by_revision for revision_id in revision_ids):
                    raise ValueError("trace_revision_unavailable")
                current_revision_id = active_revision_ids[0]
                conversation_id, turn_id = by_revision[current_revision_id]
                if route_record.chain_id is not None:
                    call_sequence = self.repository.read_next_call_sequence(
                        cursor, run_id
                    )
                    if route_record.route is ConsoleRequestRoute.AGENT_FIRST and call_sequence != 0:
                        # Only the exact owned recovery path may reuse a primary
                        # reservation. A cold invocation is not its continuation.
                        raise ValueError("trace_primary_run_already_started")
                owner = self.repository.get_attached_owner_by_conversation(
                    cursor,
                    conversation_id,
                )
                if (
                    owner is not None
                    and route_record.route is ConsoleRequestRoute.FRESH
                    and self.repository.has_unresolved_call_for_turn(
                        cursor, owner_id=owner.owner_id, turn_id=turn_id
                    )
                ):
                    raise ValueError("trace_fresh_turn_requires_owned_recovery")
                if tool_loop:
                    origin = self.repository.get_run_origin(cursor, run_id)
                    if (
                        origin is None
                        or owner is None
                        or origin.route_identity
                        != ConsoleRequestRoute.AGENT_FIRST.value
                        or (
                            origin.owner_id,
                            origin.segment_id,
                            origin.turn_id,
                            origin.policy_id,
                        )
                        != (
                            owner.owner_id,
                            owner.root_segment_id,
                            turn_id,
                            policy.policy_id,
                        )
                    ):
                        raise ValueError("trace_tool_chain_unavailable")
                    previous = self.repository.get_call_by_logical_identity(
                        cursor,
                        owner_id=owner.owner_id,
                        segment_id=owner.root_segment_id,
                        turn_id=turn_id,
                        run_id=run_id,
                        call_sequence=call_sequence - 1,
                    )
                    tail = self.repository.get_surface_tail(
                        cursor, owner.root_segment_id
                    )
                    latest_call = self.repository.get_latest_call_boundary(
                        cursor, owner.root_segment_id
                    )
                    if (
                        previous is None
                        or latest_call is None
                        or latest_call.call_id != previous.call_id
                        # Agent response settlement may still be queued when
                        # its next tool call begins; a response-bearing call
                        # already provides durable ownership for that chain.
                        or previous.state
                        not in {
                            TraceCallState.RESPONSE_STARTED,
                            TraceCallState.COMPLETE,
                        }
                        or previous.policy_id != policy.policy_id
                        or tail is None
                        or previous.surface_node_id != tail.node_id
                    ):
                        raise ValueError("trace_tool_chain_unavailable")
                if owner is None:
                    segment = self.repository.create_segment(cursor)
                    owner = self.repository.attach_owner(
                        cursor,
                        conversation_id=conversation_id,
                        root_segment_id=segment.segment_id,
                    )
                self.repository.ensure_policy(cursor, policy)
                continuation_values = tuple(
                    group.checkpoint for group in request.continuation_groups
                )
                completed_tool_turn = None
                if (
                    route_record.route
                    in {ConsoleRequestRoute.AGENT_FIRST, ConsoleRequestRoute.FRESH}
                    and len(provenance.messages_payload) >= 2
                    and all(
                        type(item) is SavedRevisionTraceProvenance
                        for item in provenance.messages_payload[-2:]
                    )
                ):
                    latest = self.repository.get_latest_call_boundary(
                        cursor, owner.root_segment_id
                    )
                    terminal = (
                        None
                        if latest is None or latest.call_id is None
                        else self.repository.get_call(cursor, latest.call_id)
                    )
                    origin = (
                        None
                        if terminal is None
                        else self.repository.get_run_origin(cursor, terminal.run_id)
                    )
                    if terminal is not None and origin is not None:
                        completed_tool_turn = CompletedToolTurnWitness(
                            origin.call_id,
                            terminal.call_id,
                            provenance.messages_payload[-2].revision_id,
                            current_revision_id,
                        )
                admission, surface_boundary = (
                    self.service.prepare_current_surface_delta(
                        cursor,
                        owner_id=owner.owner_id,
                        segment_id=owner.root_segment_id,
                        route_identity=route_identity,
                        preparation_identity=preparation_identity,
                        provenance=provenance,
                        values=tuple(request.messages_payload) + continuation_values,
                        completed_tool_turn=completed_tool_turn,
                        current_turn_id=turn_id,
                        current_policy_id=policy.policy_id,
                    )
                )
                reserved = self.repository.reserve_call(
                    cursor,
                    owner_id=owner.owner_id,
                    segment_id=owner.root_segment_id,
                    turn_id=turn_id,
                    run_id=run_id,
                    call_sequence=call_sequence,
                    idempotency_key=idempotency_key,
                    policy_id=policy.policy_id,
                )
                # Surface identity alone cannot detect a newer run with the
                # same prompt. Record the reservation in the existing ordered
                # ledger, atomically with the call, before provider dispatch.
                event_tail = self.repository.get_event_tail(
                    cursor, owner.root_segment_id
                )
                self.repository.append_event(
                    cursor,
                    segment_id=owner.root_segment_id,
                    sequence=0 if event_tail is None else event_tail.sequence + 1,
                    event_type="call_boundary",
                    call_id=reserved.call_id,
                )
            assert reserved is not None
            return ConsoleTraceCallBoundary(
                service=self.service,
                database=self.database,
                identity=TraceCallIdentity(
                    owner_id=owner.owner_id,
                    segment_id=owner.root_segment_id,
                    turn_id=turn_id,
                    run_id=run_id,
                    call_sequence=call_sequence,
                    idempotency_key=idempotency_key,
                    policy_id=policy.policy_id,
                ),
                admission=admission,
                occurred_at_factory=_utc_now,
                surface_boundary=surface_boundary,
                _reserved=reserved,
                _factory=self,
                _request=request,
                _resolution=_resolution,
                _current_revision_id=current_revision_id,
            )


__all__ = ["ConsoleTraceBoundaryFactory"]
