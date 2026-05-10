"""Local Sync v2 adapter for legacy media compatibility envelopes."""

from __future__ import annotations

from typing import Any, Callable

from tldw_chatbook.tldw_api import SyncV2Envelope

from ._helpers import call_if_present


class MediaSyncAdapter:
    """Pass media compatibility envelopes to an optional local store hook."""

    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        dataset_key: bytes,
        local_store: Any,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        del dataset_key, record_conflict
        call_if_present(local_store, "apply_media_sync_envelope", envelope)
        return {"status": "applied"}
