"""Offline-capable local normalized event producer."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from tldw_chatbook.runtime_policy.server_parity_models import NormalizedEventRecord


class LocalEventProducer:
    def __init__(self, *, source_name: str, stream_instance_id: str = "local") -> None:
        self.source_name = source_name
        self.stream_instance_id = stream_instance_id

    def emit(
        self,
        *,
        event_kind: str,
        entity_ref: Mapping[str, object],
        payload_hash: str,
        payload: Mapping[str, object] | None = None,
        event_id: str | None = None,
        emitted_at: str | None = None,
        received_at: str | None = None,
        payload_kind: str | None = None,
        server_profile_id: str | None = None,
    ) -> NormalizedEventRecord:
        if server_profile_id is not None:
            raise ValueError("Local events must not include server_profile_id")

        return NormalizedEventRecord(
            source_authority="local",
            server_profile_id=None,
            stream_name=self.source_name,
            stream_instance_id=self.stream_instance_id,
            event_kind=event_kind,
            entity_ref=entity_ref,
            payload_hash=payload_hash,
            event_id=event_id or f"local:{uuid4()}",
            emitted_at=emitted_at,
            received_at=received_at,
            transport_type="local_producer",
            payload_kind=payload_kind,
            payload=payload or {},
        )
