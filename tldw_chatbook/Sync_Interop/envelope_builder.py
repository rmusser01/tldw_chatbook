"""Build local Chatbook changes into Sync v2 envelopes."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from tldw_chatbook.Sync_Interop.crypto import encrypt_sync_payload
from tldw_chatbook.tldw_api import SyncV2Envelope


class SyncEnvelopeBuilder:
    """Small builder for first-pass local-first Sync v2 envelopes."""

    def __init__(
        self,
        *,
        dataset_id: str,
        device_id: str,
        dataset_key: bytes,
        adapter_version: int = 1,
    ) -> None:
        self.dataset_id = dataset_id
        self.device_id = device_id
        self.dataset_key = dataset_key
        self.adapter_version = adapter_version

    def build_note_upsert(
        self,
        *,
        note_id: str,
        title: str,
        body: str,
        status: str | None = None,
        tag_ids: list[str] | None = None,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
    ) -> SyncV2Envelope:
        payload = {"body": body, "title": title}
        clear: dict[str, Any] = {}
        if status is not None:
            clear["status"] = status
        if tag_ids is not None:
            clear["tag_ids"] = list(tag_ids)
        return self._encrypted_envelope(
            domain="notes",
            entity_id=note_id,
            operation="upsert",
            stable_key=note_id,
            payload=payload,
            payload_clear=clear,
            routing_metadata={"entity_kind": "note"},
            base_version=base_version,
            entity_version=entity_version,
        )

    def build_note_metadata_update(
        self,
        *,
        note_id: str,
        status: str | None = None,
        tag_ids: list[str] | None = None,
        entity_version: str | int | None = None,
    ) -> SyncV2Envelope:
        clear: dict[str, Any] = {}
        if status is not None:
            clear["status"] = status
        if tag_ids is not None:
            clear["tag_ids"] = list(tag_ids)
        return self._clear_envelope(
            domain="notes",
            entity_id=note_id,
            operation="upsert",
            stable_key=note_id,
            payload_clear=clear,
            routing_metadata={"entity_kind": "note", "record_type": "metadata"},
            payload_hash=self._payload_hash(clear),
            entity_version=entity_version,
        )

    def build_note_delete(
        self,
        *,
        note_id: str,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
    ) -> SyncV2Envelope:
        clear = {"deleted": True}
        return self._clear_envelope(
            domain="notes",
            entity_id=note_id,
            operation="delete",
            stable_key=note_id,
            payload_clear=clear,
            routing_metadata={"entity_kind": "note"},
            payload_hash=self._payload_hash(clear),
            base_version=base_version,
            entity_version=entity_version,
        )

    def build_chat_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
    ) -> SyncV2Envelope:
        stable_key = f"{conversation_id}:{message_id}"
        return self._encrypted_envelope(
            domain="chat",
            entity_id=message_id,
            operation="upsert",
            stable_key=stable_key,
            payload={"content": content, "role": role},
            payload_clear={},
            routing_metadata={"conversation_id": conversation_id, "entity_kind": "message"},
            base_version=base_version,
            entity_version=entity_version,
        )

    def build_workspace_source_ref(
        self,
        *,
        workspace_id: str,
        source_id: str,
        operation: Literal["link", "unlink"],
    ) -> SyncV2Envelope:
        clear = {"workspace_id": workspace_id, "source_id": source_id}
        return self._clear_envelope(
            domain="workspaces",
            entity_id=f"{workspace_id}:{source_id}",
            operation=operation,
            stable_key=f"{workspace_id}:{source_id}",
            payload_clear=clear,
            routing_metadata={"entity_kind": "workspace_source"},
            payload_hash=self._payload_hash(clear),
        )

    def build_source_cache(
        self,
        *,
        source_id: str,
        content_hash: str,
        cache_kind: str,
        content: str,
    ) -> SyncV2Envelope:
        stable_key = f"{source_id}:{content_hash}"
        return self._encrypted_envelope(
            domain="source_cache",
            entity_id=stable_key,
            operation="upsert",
            stable_key=stable_key,
            payload={"content": content},
            payload_clear={
                "source_id": source_id,
                "payload_hash": content_hash,
                "record_type": cache_kind,
            },
            routing_metadata={"entity_kind": "source_cache"},
        )

    def build_media_compat(
        self,
        *,
        entity_id: str,
        operation: Literal["upsert", "delete"],
        metadata: Mapping[str, Any],
    ) -> SyncV2Envelope:
        return self._clear_envelope(
            domain="media",
            entity_id=entity_id,
            operation=operation,
            stable_key=entity_id,
            payload_clear=dict(metadata),
            routing_metadata={"entity_kind": "media"},
            payload_hash=self._payload_hash(metadata),
            encryption_policy="server_trusted",
        )

    def _encrypted_envelope(
        self,
        *,
        domain: str,
        entity_id: str,
        operation: str,
        stable_key: str,
        payload: Mapping[str, Any],
        payload_clear: Mapping[str, Any],
        routing_metadata: Mapping[str, Any],
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
    ) -> SyncV2Envelope:
        encrypted = encrypt_sync_payload(payload, key=self.dataset_key)
        payload_hash = self._payload_hash(payload)
        return SyncV2Envelope(
            client_envelope_id=f"{self.device_id}:{domain}:{stable_key}:{payload_hash}",
            dataset_id=self.dataset_id,
            device_id=self.device_id,
            domain=domain,
            entity_id=entity_id,
            operation=operation,
            adapter_version=self.adapter_version,
            stable_key=stable_key,
            base_version=base_version,
            entity_version=entity_version or payload_hash,
            routing_metadata=dict(routing_metadata),
            payload_ciphertext=encrypted.model_dump_json(),
            payload_clear=dict(payload_clear),
            payload_hash=payload_hash,
            payload_size_bytes=len(encrypted.ciphertext),
        )

    def _clear_envelope(
        self,
        *,
        domain: str,
        entity_id: str,
        operation: str,
        stable_key: str,
        payload_clear: Mapping[str, Any],
        routing_metadata: Mapping[str, Any],
        payload_hash: str,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
        encryption_policy: str = "client_private_v1",
    ) -> SyncV2Envelope:
        return SyncV2Envelope(
            client_envelope_id=f"{self.device_id}:{domain}:{stable_key}:{payload_hash}",
            dataset_id=self.dataset_id,
            device_id=self.device_id,
            domain=domain,
            entity_id=entity_id,
            operation=operation,
            adapter_version=self.adapter_version,
            stable_key=stable_key,
            base_version=base_version,
            entity_version=entity_version or payload_hash,
            routing_metadata=dict(routing_metadata),
            payload_clear=dict(payload_clear),
            payload_hash=payload_hash,
            encryption_policy=encryption_policy,
        )

    @staticmethod
    def _payload_hash(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
