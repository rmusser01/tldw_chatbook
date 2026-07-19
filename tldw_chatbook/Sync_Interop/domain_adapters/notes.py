"""Local Sync v2 adapter for notes."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope

from ._helpers import call_if_present, decrypt_envelope_payload


class NotesSyncAdapter:
    """Apply note metadata and encrypted note content envelopes."""

    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        dataset_key: bytes,
        local_store: Any,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        if envelope.operation == "delete":
            call_if_present(local_store, "delete_note", envelope.entity_id)
            return {"status": "applied"}

        current_hash = call_if_present(local_store, "get_note_content_hash", envelope.entity_id)
        has_content = bool(envelope.payload_ciphertext)
        if has_content and current_hash and envelope.base_version and str(current_hash) != str(envelope.base_version):
            return record_conflict(
                envelope,
                conflict_type="encrypted_content_edit",
                message="Local note content diverged from the envelope base version.",
            )

        if envelope.payload_clear:
            call_if_present(local_store, "upsert_note_metadata", envelope.entity_id, dict(envelope.payload_clear))
        if has_content:
            payload = decrypt_envelope_payload(envelope, dataset_key=dataset_key)
            call_if_present(local_store, "upsert_note_content", envelope.entity_id, payload, envelope.payload_hash)
        return {"status": "applied"}
