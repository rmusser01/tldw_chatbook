"""Produce Sync v2 outbox envelopes for local Chat mutations."""

from __future__ import annotations

from typing import Any, Mapping

from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.sync_state import is_local_first_sync_profile_mode


class ChatSyncV2OutboxProducer:
    """Convert successful local Chat writes into durable Sync v2 outbox entries."""

    def __init__(
        self,
        *,
        state_repository: Any,
        dataset_keys: Mapping[str, bytes] | None = None,
    ) -> None:
        self.state_repository = state_repository
        self.dataset_keys = dict(dataset_keys or {})

    def enqueue_chat_message(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        parent_message_id: str | None = None,
        sequence: int | None = None,
        variant_turn_id: str | None = None,
        variant_index: int | None = None,
        variant_count: int | None = None,
        selected_variant_id: str | None = None,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
    ) -> dict[str, Any]:
        """Persist an encrypted Chat message envelope when Sync v2 is ready."""

        profile = self._sync_ready_profile(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile["status"] != "ready":
            return profile

        envelope = self._builder(profile).build_chat_message(
            conversation_id=conversation_id,
            message_id=message_id,
            role=role,
            content=content,
            parent_message_id=parent_message_id,
            sequence=sequence,
            variant_turn_id=variant_turn_id,
            variant_index=variant_index,
            variant_count=variant_count,
            selected_variant_id=selected_variant_id,
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
