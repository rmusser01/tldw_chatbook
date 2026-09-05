"""Production ownership for normalized Console provider-call boundaries."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone

from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
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
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceCallBoundary,
    ConsoleTraceService,
    TraceCallIdentity,
)

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
            _saved_revision_ids(
                provenance.messages_payload[active_descriptor_index]
            )
        )
        if len(active_revision_ids) != 1:
            raise ValueError("trace_turn_unavailable")
        revision_ids = message_revision_ids + tuple(
            revision_id
            for descriptor in provenance.continuations
            for revision_id in _saved_revision_ids(descriptor)
        )
        policy = frozen_policy_from_provenance(semantic_provenance)
        preparation_identity = new_opaque_id()
        run_id = route_record.chain_id or new_opaque_id()
        idempotency_key = new_opaque_id()
        call_sequence = 0
        reserved = None
        with self._lock:
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
                by_revision = {
                    str(row[0]): (str(row[1]), str(row[2])) for row in rows
                }
                if any(revision_id not in by_revision for revision_id in revision_ids):
                    raise ValueError("trace_revision_unavailable")
                current_revision_id = active_revision_ids[0]
                conversation_id, turn_id = by_revision[current_revision_id]
                owner = self.repository.get_attached_owner_by_conversation(
                    cursor,
                    conversation_id,
                )
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
                admission, surface_boundary = self.service.prepare_current_surface_delta(
                    cursor,
                    owner_id=owner.owner_id,
                    segment_id=owner.root_segment_id,
                    route_identity=route_identity,
                    preparation_identity=preparation_identity,
                    provenance=provenance,
                    values=tuple(request.messages_payload) + continuation_values,
                )
                if route_record.chain_id is not None:
                    call_sequence = self.repository.read_next_call_sequence(
                        cursor,
                        run_id,
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
            )


__all__ = ["ConsoleTraceBoundaryFactory"]
