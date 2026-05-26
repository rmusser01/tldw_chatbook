"""Produce Sync v2 outbox envelopes for local Notes mutations."""

from __future__ import annotations

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
    ) -> None:
        self.state_repository = state_repository
        self.dataset_keys = dict(dataset_keys or {})

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

        Returns:
            A status mapping. Enqueued results include the durable outbox entry;
            skipped results include a reason describing the missing prerequisite.
        """

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
    ) -> dict[str, Any]:
        """Persist a Notes delete tombstone envelope when Sync v2 is ready.

        Args:
            server_profile_id: Server profile that owns the outbox source scope.
            authenticated_principal_id: Optional authenticated principal scope.
            workspace_scope: Optional workspace scope for scoped outbox entries.
            note_id: Local Notes entity ID.
            base_version: Optional source entity version before the delete.
            entity_version: Optional source entity version after the delete.

        Returns:
            A status mapping. Enqueued results include the durable outbox entry;
            skipped results include a reason describing the missing prerequisite.
        """

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
        return self._enqueue(
            profile=profile,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            envelope=envelope,
        )

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
