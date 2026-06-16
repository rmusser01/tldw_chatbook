"""Server-trusted M1 apply adapter for notes.note (cleartext, idempotent)."""
from __future__ import annotations

from html import escape as html_escape
from typing import Any, Callable

from tldw_chatbook.Utils.input_validation import sanitize_string
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
        if notes_mirror is None or dataset_id is None:
            raise ValueError("notes_mirror and dataset_id must be provided to NotesM1SyncAdapter")
        object_id = envelope.object_id or envelope.entity_id
        if object_id is None:
            return record_conflict(
                envelope,
                conflict_type="missing_object_id",
                message="notes.note envelopes require object_id or entity_id.",
            )
        cursor = envelope.server_cursor or 0
        revision = envelope.object_revision or 0
        payload_hash = envelope.payload_hash or ""
        existing = notes_mirror.get(dataset_id, object_id)
        if existing is not None:
            if existing.object_revision > revision:
                # Stale envelope (older revision than local) — drop, do not overwrite.
                return {"status": "noop", "object_id": object_id}
            if existing.object_revision == revision and existing.object_hash == payload_hash:
                # Exact duplicate already applied.
                return {"status": "noop", "object_id": object_id}

        if envelope.operation == "tombstone" or envelope.deleted:
            local_store.soft_delete_note(object_id, object_revision=revision)
        else:
            payload = self._validated_note_payload(envelope.payload)
            if payload is None:
                return record_conflict(
                    envelope,
                    conflict_type="invalid_notes_payload",
                    message="notes.note upsert payload requires non-empty title and content strings.",
                )
            local_store.upsert_note(object_id, payload, object_revision=revision)
        notes_mirror.record(
            dataset_id, object_id,
            object_revision=revision, object_hash=payload_hash, server_cursor=cursor,
        )
        return {"status": "applied", "object_id": object_id}

    @staticmethod
    def _validated_note_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
        title = payload.get("title")
        content = payload.get("content", payload.get("body"))
        if not isinstance(title, str) or not isinstance(content, str):
            return None
        title = sanitize_string(title, max_length=500).strip()
        content = sanitize_string(content, max_length=100_000)
        if not title or not content.strip():
            return None
        return {
            "title": html_escape(title, quote=False),
            "content": html_escape(content, quote=False),
        }
