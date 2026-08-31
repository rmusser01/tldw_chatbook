"""Canonical Personal Context whole-object adapter for Sync v2."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    RecordState,
    SyncMode,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_chatbook.tldw_api import SyncV2Envelope
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.crypto import (
    decrypt_sync_payload,
    encrypt_sync_payload,
)


_MAX_OBJECT_BYTES = 16_384
_MODELS = {
    "manifest": ProfileManifest,
    "scope": ProfileScope,
    "record": ProfileRecord,
    "proposal": ProfileProposal,
}


class PersonalContextSyncValidationError(ValueError):
    """Fail closed with one stable content-free reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class PersonalContextSyncAdapter:
    """Build and apply schema-v1 Personal Context Sync envelopes."""

    def __init__(self, *, integrity_key: bytes, integrity_key_id: str) -> None:
        if len(integrity_key) != 32:
            raise ValueError("Personal Context integrity key must be 32 bytes")
        if not isinstance(integrity_key_id, str) or not integrity_key_id:
            raise ValueError("Personal Context integrity key id is required")
        self._key = bytes(integrity_key)
        self._key_id = integrity_key_id

    def build_envelope(
        self,
        *,
        entry: Any,
        body: Mapping[str, Any],
        dataset_id: str,
        device_id: str,
        sync_head: Mapping[str, Any] | None = None,
    ) -> SyncV2Envelope:
        """Convert one authenticated outbox snapshot into an exact envelope."""

        object_type, payload = _parse_body(entry.object_type, body)
        identity = _identity(object_type, payload)
        if entry.object_id != identity["object_id"]:
            raise PersonalContextSyncValidationError("personal_context_identity_conflict")
        canonical = canonical_json_bytes(payload)
        if len(canonical) > _MAX_OBJECT_BYTES:
            raise PersonalContextSyncValidationError("personal_context_payload_too_large")
        operation = identity["operation"]
        return SyncEnvelopeBuilder.build_personal_context_whole_object(
            client_envelope_id=f"personal-context:{entry.outbox_id}",
            dataset_id=dataset_id,
            domain=f"personal_context.{object_type}",
            object_id=identity["object_id"],
            parent_id=identity["parent_id"],
            operation=operation,
            device_id=device_id,
            base_version=identity["base_version"],
            entity_version=identity["entity_version"],
            object_revision=identity["object_revision"],
            base_server_cursor=(
                None if sync_head is None else sync_head.get("server_cursor")
            ),
            base_object_revision=(
                None if sync_head is None else sync_head.get("object_revision")
            ),
            base_object_hash=(
                None if sync_head is None else sync_head.get("payload_hash")
            ),
            integrity_key_id=self._key_id,
            profile_id=identity["profile_id"],
            purge_generation=identity["purge_generation"],
            payload=payload,
            payload_hash=self._tag(canonical),
            payload_size_bytes=len(canonical),
        )

    @staticmethod
    def protect_for_storage(
        envelope: SyncV2Envelope,
        *,
        storage_key: bytes,
    ) -> SyncV2Envelope:
        """Seal clear Personal Context content before generic Sync persistence."""

        if not envelope.domain.startswith("personal_context."):
            raise PersonalContextSyncValidationError(
                "personal_context_domain_mismatch"
            )
        encrypted = encrypt_sync_payload(envelope.payload, key=storage_key)
        values = envelope.model_dump(mode="json")
        values.update(
            {
                "payload": {},
                "payload_clear": {},
                "payload_ciphertext": encrypted.model_dump_json(),
                "encryption_metadata": {
                    **envelope.encryption_metadata,
                    "personal_context_local_staging": "dataset-key-v1",
                },
            }
        )
        return SyncV2Envelope.model_validate(values)

    @staticmethod
    def restore_from_storage(
        envelope: SyncV2Envelope,
        *,
        storage_key: bytes,
    ) -> SyncV2Envelope:
        """Authenticate and restore one locally staged Personal Context body."""

        marker = envelope.encryption_metadata.get(
            "personal_context_local_staging"
        )
        if marker != "dataset-key-v1" or not envelope.payload_ciphertext:
            raise PersonalContextSyncValidationError(
                "personal_context_storage_invalid"
            )
        try:
            encrypted = json.loads(envelope.payload_ciphertext)
            payload = decrypt_sync_payload(encrypted, key=storage_key)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PersonalContextSyncValidationError(
                "personal_context_storage_invalid"
            ) from exc
        metadata = dict(envelope.encryption_metadata)
        metadata.pop("personal_context_local_staging", None)
        values = envelope.model_dump(mode="json")
        values.update(
            {
                "payload": payload,
                "payload_clear": payload,
                "payload_ciphertext": None,
                "encryption_metadata": metadata,
            }
        )
        return SyncV2Envelope.model_validate(values)

    def apply_inbound(self, envelope: SyncV2Envelope, *, service: Any) -> Any:
        """Validate one pulled whole object and invoke only its owner service."""

        if (
            envelope.adapter_version != 1
            or envelope.schema_version != 1
            or not envelope.domain.startswith("personal_context.")
            or envelope.encryption_policy != "server_trusted_v1"
        ):
            raise PersonalContextSyncValidationError(
                "personal_context_schema_unsupported"
            )
        if envelope.routing_metadata.get("integrity_key_id") != self._key_id:
            raise PersonalContextSyncValidationError(
                "personal_context_integrity_key_invalid"
            )
        canonical = canonical_json_bytes(envelope.payload)
        if len(canonical) > _MAX_OBJECT_BYTES:
            raise PersonalContextSyncValidationError(
                "personal_context_payload_too_large"
            )
        if not hmac.compare_digest(envelope.payload_hash, self._tag(canonical)):
            raise PersonalContextSyncValidationError(
                "personal_context_integrity_invalid"
            )
        object_type = envelope.domain.removeprefix("personal_context.")
        try:
            parsed_type, value = _parse_payload(object_type, envelope.payload)
            identity = _identity(parsed_type, envelope.payload)
        except (ValidationError, ValueError, TypeError) as exc:
            raise PersonalContextSyncValidationError(
                "personal_context_payload_invalid"
            ) from exc
        if (
            envelope.object_id != identity["object_id"]
            or envelope.parent_id != identity["parent_id"]
            or envelope.operation != identity["operation"]
            or not _same_wire_version(
                envelope.entity_version,
                identity["entity_version"],
            )
        ):
            raise PersonalContextSyncValidationError(
                "personal_context_identity_conflict"
            )
        return service.apply_sync_object(
            domain=envelope.domain,
            value=value,
            actor_type="sync",
            actor_id=envelope.device_id,
            base_object_hash=envelope.base_object_hash,
        )

    def _tag(self, canonical: bytes) -> str:
        return "hmac-sha256-v1:" + hmac.new(
            self._key, canonical, hashlib.sha256
        ).hexdigest()


def _parse_body(
    expected_type: str,
    body: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(body, Mapping):
        raise PersonalContextSyncValidationError("invalid_canonical_object")
    if body.get("version") != 1 or set(body) != {"version", expected_type}:
        raise PersonalContextSyncValidationError("invalid_canonical_object")
    value = body.get(expected_type)
    if not isinstance(value, Mapping):
        raise PersonalContextSyncValidationError("invalid_canonical_object")
    try:
        parsed_type, parsed = _parse_payload(expected_type, value)
    except (ValidationError, ValueError, TypeError) as exc:
        raise PersonalContextSyncValidationError("invalid_canonical_object") from exc
    payload = (
        parsed.model_dump(mode="json")
        if hasattr(parsed, "model_dump")
        else dict(parsed)
    )
    return parsed_type, payload


def _parse_payload(object_type: str, value: Mapping[str, Any]) -> tuple[str, Any]:
    model = _MODELS.get(object_type)
    if model is not None:
        parsed = model.model_validate(value)
        if object_type == "record" and parsed.controls.sync_mode is SyncMode.DEVICE_ONLY:
            raise ValueError("Device-only records cannot synchronize")
        if (
            object_type == "proposal"
            and parsed.state.value == "pending"
            and parsed.proposed_record is not None
            and parsed.proposed_record.controls.sync_mode is SyncMode.DEVICE_ONLY
        ):
            raise ValueError("Device-only proposals cannot synchronize")
        return object_type, parsed
    if object_type != "purge" or set(value) != {
        "schema_version",
        "profile_id",
        "purge_generation",
    }:
        raise ValueError("Unsupported Personal Context object")
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("profile_id"), str)
        or type(value.get("purge_generation")) is not int
        or value["purge_generation"] < 1
    ):
        raise ValueError("Invalid Personal Context purge barrier")
    return object_type, dict(value)


def _identity(object_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    _parsed_type, value = _parse_payload(object_type, payload)
    if object_type == "manifest":
        return {
            "object_id": value.profile_id,
            "parent_id": None,
            "profile_id": value.profile_id,
            "base_version": None,
            "entity_version": value.current_version_id,
            "object_revision": value.revision,
            "operation": "upsert",
            "purge_generation": value.purge_generation,
        }
    if object_type == "scope":
        return {
            "object_id": value.scope_id,
            "parent_id": value.profile_id,
            "profile_id": value.profile_id,
            "base_version": None,
            "entity_version": value.version_id,
            "object_revision": None,
            "operation": "upsert",
            "purge_generation": None,
        }
    if object_type == "record":
        return {
            "object_id": value.record_id,
            "parent_id": value.scope_id,
            "profile_id": value.profile_id,
            "base_version": value.parent_version_id,
            "entity_version": value.version_id,
            "object_revision": None,
            "operation": (
                "tombstone" if value.state is RecordState.DELETED else "upsert"
            ),
            "purge_generation": None,
        }
    if object_type == "proposal":
        proposal_version = "sync-proposal-sha256:" + hashlib.sha256(
            canonical_json_bytes(value.model_dump(mode="json"))
        ).hexdigest()
        return {
            "object_id": value.proposal_id,
            "parent_id": value.scope_id,
            "profile_id": value.profile_id,
            "base_version": value.base_version_id,
            "entity_version": proposal_version,
            "object_revision": None,
            "operation": "upsert",
            "purge_generation": None,
        }
    return {
        "object_id": value["profile_id"],
        "parent_id": None,
        "profile_id": value["profile_id"],
        "base_version": None,
        "entity_version": value["purge_generation"],
        "object_revision": value["purge_generation"],
        "operation": "tombstone",
        "purge_generation": value["purge_generation"],
    }


def _same_wire_version(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


__all__ = ["PersonalContextSyncAdapter", "PersonalContextSyncValidationError"]
