"""Local Sync v2 adapter for source cache records."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope

from ._helpers import call_if_present, decrypt_envelope_payload


class SourceCacheSyncAdapter:
    """Apply source-cache entries by source ID plus content hash."""

    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        dataset_key: bytes,
        local_store: Any,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        del record_conflict
        stable_key = envelope.stable_key or envelope.entity_id
        payload = decrypt_envelope_payload(envelope, dataset_key=dataset_key)
        call_if_present(local_store, "upsert_source_cache", stable_key, payload, dict(envelope.payload_clear))
        return {"status": "applied"}
