"""Produce Sync v2 outbox envelopes for local Notes mutations."""

from __future__ import annotations

import json
from typing import Any, Mapping

from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.sync_state import is_local_first_sync_profile_mode


class NotesSyncV2OutboxProducer:
    """Convert successful local Notes writes into durable Sync v2 outbox entries.

    Args:
        state_repository: Repository that owns Sync v2 profile state and durable
            local outbox persistence.
        dataset_keys: Mapping of dataset IDs to in-memory dataset keys used to
            encrypt private Notes payloads before persistence.
    """

    def __init__(
        self,
        *,
        state_repository: Any,
        dataset_keys: Mapping[str, bytes] | None = None,
        notes_db: Any | None = None,
    ) -> None:
        self.state_repository = state_repository
        self.dataset_keys = dataset_keys if dataset_keys is not None else {}
        self.notes_db = notes_db

    @staticmethod
    def build_organization_envelope(
        intent: Mapping[str, Any], *, device_id: str
    ) -> dict[str, Any]:
        """Build the clear server-trusted envelope from one immutable intent row.

        Organization payloads deliberately do not pass through the private note
        builder: transport protection and server at-rest protection are the
        authoritative controls for this non-content metadata.
        """

        payload = json.loads(str(intent["payload_json"]))
        dependencies = json.loads(str(intent["dependency_refs_json"]))
        return {
            "client_envelope_id": str(intent["intent_id"]),
            "dataset_id": str(intent["dataset_id"]),
            "device_id": device_id,
            "domain": str(intent["domain"]),
            "object_id": str(intent["object_id"]),
            "operation": str(intent["operation"]),
            "adapter_version": 1,
            "schema_version": int(intent["schema_version"]),
            "object_revision": int(intent["source_version"]),
            "base_object_revision": intent["base_object_revision"],
            "base_object_hash": intent["base_object_hash"],
            "dependencies": dependencies,
            "deleted": str(intent["operation"]) == "tombstone",
            "payload": payload,
            "payload_clear": payload,
            "payload_hash": str(intent["payload_hash"]),
            "encryption_policy": "server_trusted_v1",
            "encryption_metadata": {"policy": "server_trusted_v1"},
        }

    def enqueue_note_upsert(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        note_id: str,
        title: str,
        content: str,
        status: str | None = None,
        tag_ids: list[str] | None = None,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
        publication_intent_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an encrypted Notes upsert envelope when Sync v2 is ready.

        Args:
            server_profile_id: Server profile that owns the outbox source scope.
            authenticated_principal_id: Optional authenticated principal scope.
            workspace_scope: Optional workspace scope for scoped outbox entries.
            note_id: Local Notes entity ID.
            title: Note title to encrypt into the payload.
            content: Note body to encrypt into the payload.
            status: Optional clear-text note status metadata.
            tag_ids: Optional clear-text tag identifiers for routing/metadata.
            base_version: Optional source entity version before the mutation.
            entity_version: Optional source entity version after the mutation.
            publication_intent_id: Optional immutable Notes publication intent
                identity. When supplied, it owns transport idempotency instead
                of the mutable note payload hash.

        Returns:
            A status mapping. Enqueued results include the durable outbox entry;
            skipped results include a reason describing the missing prerequisite.
        """

        blocked = self._blocking_receipt(note_id)
        if blocked is not None:
            return blocked
        profile = self._sync_ready_profile(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile["status"] != "ready":
            return profile

        builder = self._builder(profile)
        envelope = builder.build_note_upsert(
            note_id=note_id,
            title=title,
            body=content,
            status=status,
            tag_ids=tag_ids,
            base_version=base_version,
            entity_version=entity_version,
        )
        if publication_intent_id is not None:
            envelope = envelope.model_copy(
                update={"client_envelope_id": str(publication_intent_id)}
            )
        return self._enqueue(
            profile=profile,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            envelope=envelope,
        )

    def enqueue_note_delete(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        note_id: str,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
        publication_intent_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a Notes delete tombstone envelope when Sync v2 is ready.

        Args:
            server_profile_id: Server profile that owns the outbox source scope.
            authenticated_principal_id: Optional authenticated principal scope.
            workspace_scope: Optional workspace scope for scoped outbox entries.
            note_id: Local Notes entity ID.
            base_version: Optional source entity version before the delete.
            entity_version: Optional source entity version after the delete.
            publication_intent_id: Optional immutable Notes publication intent
                identity used for transport idempotency.

        Returns:
            A status mapping. Enqueued results include the durable outbox entry;
            skipped results include a reason describing the missing prerequisite.
        """

        blocked = self._blocking_receipt(note_id)
        if blocked is not None:
            return blocked
        profile = self._sync_ready_profile(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile["status"] != "ready":
            return profile

        envelope = self._builder(profile).build_note_delete(
            note_id=note_id,
            base_version=base_version,
            entity_version=entity_version,
        )
        if publication_intent_id is not None:
            envelope = envelope.model_copy(
                update={"client_envelope_id": str(publication_intent_id)}
            )
        return self._enqueue(
            profile=profile,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            envelope=envelope,
        )

    def _blocking_receipt(self, note_id: str) -> dict[str, str] | None:
        """Fail closed when the Notes owner marks a note pending organization."""

        if self.notes_db is None:
            return {"status": "skipped", "reason": "notes_authority_unavailable"}
        if self.notes_db.is_note_dispatchable(note_id):
            return None
        return {"status": "skipped", "reason": "pending_organization"}

    def _sync_ready_profile(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> dict[str, Any]:
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            return {"status": "skipped", "reason": "profile_not_configured"}
        if not is_local_first_sync_profile_mode(profile.get("profile_mode")):
            return {"status": "skipped", "reason": "profile_not_local_first"}

        device_id = profile.get("device_id")
        dataset_id = profile.get("dataset_id")
        if not device_id or not dataset_id:
            return {"status": "skipped", "reason": "profile_missing_dataset_identity"}
        dataset_key = self.dataset_keys.get(str(dataset_id))
        if dataset_key is None:
            return {"status": "skipped", "reason": "dataset_key_unavailable"}

        return {
            "status": "ready",
            "device_id": str(device_id),
            "dataset_id": str(dataset_id),
            "dataset_key": dataset_key,
        }

    @staticmethod
    def _builder(profile: Mapping[str, Any]) -> SyncEnvelopeBuilder:
        return SyncEnvelopeBuilder(
            dataset_id=str(profile["dataset_id"]),
            device_id=str(profile["device_id"]),
            dataset_key=profile["dataset_key"],
        )

    def _enqueue(
        self,
        *,
        profile: Mapping[str, Any],
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        envelope: Any,
    ) -> dict[str, Any]:
        entry = self.state_repository.enqueue_sync_v2_outbox_envelope(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=str(profile["dataset_id"]),
            envelope=envelope,
        )
        return {"status": "enqueued", "outbox_entry": entry}
