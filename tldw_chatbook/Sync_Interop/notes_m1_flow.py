"""Focused notes.note push/pull/apply flow against an M1 Sync v2 server."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope



class NotesM1SyncFlow:
    def __init__(self, *, client, builder, mirror, local_store, dataset_id: str, device_id: str) -> None:
        self.client = client
        self.builder = builder
        self.mirror = mirror
        self.local_store = local_store
        self.dataset_id = dataset_id
        self.device_id = device_id
        self._client_sequence = 0

    async def push(self, envelopes: list[SyncV2Envelope]) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from tldw_chatbook.tldw_api import SyncV2PushRequest

        for env in envelopes:
            self._client_sequence += 1
            env.client_sequence = self._client_sequence
        request = SyncV2PushRequest(dataset_id=self.dataset_id, device_id=self.device_id, envelopes=envelopes)
        response = await self.client.push_sync_v2_envelopes(request)
        payload_hashes = {env.client_envelope_id: env.payload_hash for env in envelopes}
        mirror_errors = 0
        for acknowledgement in [
            *response.accepted,
            *(getattr(response, "idempotent", []) or []),
        ]:
            if not self._record_acknowledgement(acknowledgement, payload_hashes):
                mirror_errors += 1
        return {
            "accepted": len(response.accepted),
            "idempotent": len(getattr(response, "idempotent", []) or []),
            "rejected": len(getattr(response, "rejected", []) or []),
            "conflicts": len(getattr(response, "conflicts", []) or []),
            "apply_errors": len(getattr(response, "apply_errors", []) or []),
            "mirror_errors": mirror_errors,
            "next_cursor": response.next_cursor,
        }

    async def pull(self, *, cursor: int) -> dict[str, Any]:
        response = await self.client.pull_sync_v2_envelopes(
            dataset_id=self.dataset_id, device_id=self.device_id, cursor=str(cursor), domains=["notes.note"],
        )
        applier = SyncEnvelopeApplier(local_store=self.local_store, notes_mirror=self.mirror, dataset_id=self.dataset_id)
        applied = noop = conflicts = 0
        for env in response.envelopes:
            status = applier.apply(env).get("status")
            if status == "applied":
                applied += 1
            elif status == "noop":
                noop += 1
            else:
                conflicts += 1
        return {
            "applied": applied,
            "noop": noop,
            "conflicts": conflicts,
            "next_cursor": response.next_cursor,
            "has_more": response.has_more,
        }

    def _record_acknowledgement(self, acknowledgement: Any, payload_hashes: dict[str, str]) -> bool:
        object_id = getattr(acknowledgement, "object_id", None) or getattr(acknowledgement, "entity_id", None)
        if object_id is None:
            return False

        existing = self.mirror.get(self.dataset_id, object_id)
        object_revision = getattr(acknowledgement, "object_revision", None)
        server_cursor = getattr(acknowledgement, "server_cursor", None)
        if server_cursor is None:
            server_cursor = getattr(acknowledgement, "server_sequence", None)

        payload_hash = payload_hashes.get(getattr(acknowledgement, "client_envelope_id", ""))
        if existing is not None and (object_revision is None or server_cursor is None):
            object_revision = existing.object_revision if object_revision is None else max(existing.object_revision, object_revision)
            server_cursor = existing.server_cursor if server_cursor is None else max(existing.server_cursor, server_cursor)
            payload_hash = existing.object_hash
        elif payload_hash is None or object_revision is None or server_cursor is None:
            return False
        elif existing is not None:
            object_revision = max(existing.object_revision, object_revision)
            server_cursor = max(existing.server_cursor, server_cursor)

        self.mirror.record(
            self.dataset_id,
            object_id,
            object_revision=object_revision,
            object_hash=payload_hash,
            server_cursor=server_cursor,
        )
        return True
