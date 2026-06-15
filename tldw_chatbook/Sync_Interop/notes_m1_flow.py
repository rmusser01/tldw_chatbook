"""Focused notes.note push/pull/apply flow against an M1 Sync v2 server."""
from __future__ import annotations

from typing import Any

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.tldw_api import SyncV2Envelope, SyncV2PushRequest


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
        for env in envelopes:
            self._client_sequence += 1
            env.client_sequence = self._client_sequence
        request = SyncV2PushRequest(dataset_id=self.dataset_id, device_id=self.device_id, envelopes=envelopes)
        response = await self.client.push_sync_v2_envelopes(request)
        for accepted in response.accepted:
            oid = accepted.object_id
            if oid is None:
                continue
            self.mirror.record(
                self.dataset_id, oid,
                object_revision=accepted.object_revision or 0,
                object_hash=self._hash_for(envelopes, accepted.client_envelope_id),
                server_cursor=accepted.server_cursor or 0,
            )
        return {
            "accepted": len(response.accepted),
            "idempotent": len(getattr(response, "idempotent", []) or []),
            "rejected": len(response.rejected),
            "conflicts": len(response.conflicts),
            "apply_errors": len(getattr(response, "apply_errors", []) or []),
            "next_cursor": response.next_cursor,
        }

    async def pull(self, *, cursor: int) -> dict[str, Any]:
        response = await self.client.pull_sync_v2_envelopes(
            dataset_id=self.dataset_id, device_id=self.device_id, cursor=str(cursor), domains=["notes.note"],
        )
        applier = SyncEnvelopeApplier(local_store=self.local_store, notes_mirror=self.mirror, dataset_id=self.dataset_id)
        applied = 0
        for env in response.envelopes:
            result = applier.apply(env)
            if result.get("status") == "applied":
                applied += 1
        return {"applied": applied, "next_cursor": response.next_cursor, "has_more": response.has_more}

    @staticmethod
    def _hash_for(envelopes: list[SyncV2Envelope], client_envelope_id: str) -> str:
        for env in envelopes:
            if env.client_envelope_id == client_envelope_id:
                return env.payload_hash
        return ""
