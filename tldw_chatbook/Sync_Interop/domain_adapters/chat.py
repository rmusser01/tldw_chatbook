"""Local Sync v2 adapter for chat messages."""

from __future__ import annotations

import re
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope

from ._helpers import call_if_present, decrypt_envelope_payload


_CANONICAL_PAYLOAD_HASH = re.compile(r"sha256:[0-9a-f]{64}").fullmatch


class ChatSyncAdapter:
    """Apply versioned chat message envelopes."""

    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        dataset_key: bytes,
        local_store: Any,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        stable_key = envelope.stable_key or envelope.entity_id
        current_hash = call_if_present(local_store, "get_chat_message_hash", stable_key)
        if current_hash is not None and (
            type(current_hash) is not str
            or _CANONICAL_PAYLOAD_HASH(current_hash) is None
        ):
            return record_conflict(
                envelope,
                conflict_type="chat_message_hash_mismatch",
                message="A chat message with this stable ID has an unreadable local hash.",
            )
        if current_hash == envelope.payload_hash:
            return {"status": "noop"}
        if current_hash and current_hash != envelope.payload_hash:
            if envelope.base_version != current_hash:
                return record_conflict(
                    envelope,
                    conflict_type="chat_message_hash_mismatch",
                    message="A chat message with this stable ID already has different content.",
                )
        if envelope.operation == "delete":
            call_if_present(
                local_store,
                "delete_chat_message",
                stable_key,
                envelope.payload_hash,
            )
            return {"status": "applied"}
        payload = decrypt_envelope_payload(envelope, dataset_key=dataset_key)
        call_if_present(
            local_store,
            "append_chat_message",
            stable_key,
            payload,
            envelope.payload_hash,
        )
        return {"status": "applied"}
