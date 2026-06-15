"""Server-trusted M1 apply adapter for notes.note (cleartext, idempotent)."""
from __future__ import annotations

from typing import Any, Callable

from tldw_chatbook.tldw_api import SyncV2Envelope


class NotesM1SyncAdapter:
    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        local_store: Any,
        notes_mirror: Any,
        dataset_id: str,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        object_id = envelope.object_id or envelope.entity_id
        cursor = envelope.server_cursor or 0
        revision = envelope.object_revision or 0
        existing = notes_mirror.get(dataset_id, object_id)
        if existing is not None and existing.object_revision == revision and existing.object_hash == envelope.payload_hash:
            return {"status": "noop", "object_id": object_id}

        if envelope.operation == "tombstone" or envelope.deleted:
            local_store.soft_delete_note(object_id, object_revision=revision)
        else:
            local_store.upsert_note(object_id, dict(envelope.payload), object_revision=revision)
        notes_mirror.record(
            dataset_id, object_id,
            object_revision=revision, object_hash=envelope.payload_hash, server_cursor=cursor,
        )
        return {"status": "applied", "object_id": object_id}
