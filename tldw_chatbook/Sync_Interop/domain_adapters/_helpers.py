"""Shared helpers for Chatbook Sync v2 domain adapters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope


def decrypt_envelope_payload(envelope: SyncV2Envelope, *, dataset_key: bytes) -> dict[str, Any]:
    if not envelope.payload_ciphertext:
        return {}
    return decrypt_sync_payload(json.loads(envelope.payload_ciphertext), key=dataset_key)


def call_if_present(target: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, name, None)
    if callable(method):
        return method(*args, **kwargs)
    return None
