"""Produce Sync v2 outbox envelopes for local Chat mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.sync_state import is_local_first_sync_profile_mode


@dataclass(frozen=True, slots=True, repr=False)
class ChatSyncIntentRecord:
    """Immutable message fields proven by one committed source intent."""

    conversation_id: str
    message_id: str
    role: str
    content: str
    parent_message_id: str | None
    provider_continuation_json: str | None
    thinking_blocks_json: str | None
    assistant_generation_state: str | None
    message_version: int
    payload_hash: str
    base_payload_hash: str | None

    def __repr__(self) -> str:
        return (
            "ChatSyncIntentRecord("
            f"message_version={self.message_version}, private_fields=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class ChatSyncDeleteIntentRecord:
    """Immutable proof for one committed message tombstone."""

    conversation_id: str
    message_id: str
    message_version: int
    payload_hash: str
    base_payload_hash: str


class ChatSyncIntentSource(Protocol):
    """Read exact already-committed Chat message sync intent proof."""

    def read_committed_chat_sync_intent(
        self,
        *,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> ChatSyncIntentRecord | None:
        """Return a verified immutable source row or ``None``."""

    def read_committed_chat_delete_intent(
        self,
        *,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> ChatSyncDeleteIntentRecord | None:
        """Return a verified immutable tombstone source row or ``None``."""


class ChatSyncV2OutboxProducer:
    """Convert successful local Chat writes into durable Sync v2 outbox entries."""

    def __init__(
        self,
        *,
        state_repository: Any,
        dataset_keys: Mapping[str, bytes] | None = None,
        source: ChatSyncIntentSource | None = None,
    ) -> None:
        self.state_repository = state_repository
        self.dataset_keys = dict(dataset_keys or {})
        self.source = source

    def reconcile_chat_message_intent(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> dict[str, Any]:
        """Project one exact committed source intent into outbox plus receipt."""
        if not bool(getattr(self.state_repository, "is_durable", False)):
            return {"status": "skipped", "reason": "state_repository_not_durable"}
        profile = self._sync_ready_profile(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile["status"] != "ready":
            return profile
        if self.source is None:
            return {"status": "skipped", "reason": "source_intent_unavailable"}
        source_record = self.source.read_committed_chat_sync_intent(
            message_id=message_id,
            message_version=message_version,
            payload_hash=payload_hash,
        )
        if source_record is None:
            return {"status": "skipped", "reason": "source_intent_unavailable"}

        envelope = self._builder(profile).build_chat_message(
            conversation_id=source_record.conversation_id,
            message_id=source_record.message_id,
            role=source_record.role,
            content=source_record.content,
            parent_message_id=source_record.parent_message_id,
            provider_continuation_json=source_record.provider_continuation_json,
            thinking_blocks_json=source_record.thinking_blocks_json,
            assistant_generation_state=source_record.assistant_generation_state,
            base_version=source_record.base_payload_hash,
            entity_version=source_record.message_version,
        )
        if envelope.payload_hash != source_record.payload_hash:
            return {"status": "skipped", "reason": "source_intent_unavailable"}
        envelope = envelope.model_copy(
            update={
                "client_envelope_id": (
                    f"{envelope.client_envelope_id}:source-version:"
                    f"{source_record.message_version}"
                )
            }
        )
        projected = (
            self.state_repository.enqueue_sync_v2_outbox_envelope_with_source_receipt(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=str(profile["dataset_id"]),
                envelope=envelope,
                source_entity_id=source_record.message_id,
                source_version=source_record.message_version,
                source_payload_hash=source_record.payload_hash,
            )
        )
        return {"status": "enqueued", **projected}

    def reconcile_chat_message_delete_intent(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> dict[str, Any]:
        """Project one exact committed Chat tombstone into outbox plus receipt."""
        if not bool(getattr(self.state_repository, "is_durable", False)):
            return {"status": "skipped", "reason": "state_repository_not_durable"}
        profile = self._sync_ready_profile(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile["status"] != "ready":
            return profile
        reader = getattr(self.source, "read_committed_chat_delete_intent", None)
        if not callable(reader):
            return {"status": "skipped", "reason": "source_intent_unavailable"}
        source_record = reader(
            message_id=message_id,
            message_version=message_version,
            payload_hash=payload_hash,
        )
        if source_record is None:
            return {"status": "skipped", "reason": "source_intent_unavailable"}

        envelope = self._builder(profile).build_chat_message_delete(
            conversation_id=source_record.conversation_id,
            message_id=source_record.message_id,
            base_version=source_record.base_payload_hash,
            entity_version=source_record.message_version,
        )
        if envelope.payload_hash != source_record.payload_hash:
            return {"status": "skipped", "reason": "source_intent_unavailable"}
        envelope = envelope.model_copy(
            update={
                "client_envelope_id": (
                    f"{envelope.client_envelope_id}:source-version:"
                    f"{source_record.message_version}"
                )
            }
        )
        projected = (
            self.state_repository.enqueue_sync_v2_outbox_envelope_with_source_receipt(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=str(profile["dataset_id"]),
                envelope=envelope,
                source_entity_id=source_record.message_id,
                source_version=source_record.message_version,
                source_payload_hash=source_record.payload_hash,
                supersede_object_history=True,
            )
        )
        return {"status": "enqueued", **projected}

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
        thinking_blocks_json: str | None = None,
        assistant_generation_state: str | None = None,
        base_version: str | int | None = None,
        entity_version: str | int | None = None,
    ) -> dict[str, Any]:
        """Persist an encrypted Chat message envelope when Sync v2 is ready.

        Args:
            server_profile_id: Server profile that owns the outbox source scope.
            authenticated_principal_id: Optional authenticated principal scope.
            workspace_scope: Optional workspace scope for scoped outbox entries.
            conversation_id: Durable local conversation ID that owns the message.
            message_id: Durable local message ID.
            role: Chat role for the message, such as ``user`` or ``assistant``.
            content: Message content encrypted into the envelope payload.
            parent_message_id: Optional previous durable message ID for restore continuity.
            sequence: Optional 1-based order among sync-eligible messages.
            variant_turn_id: Optional turn ID shared by regenerated variants.
            variant_index: Optional selected variant index.
            variant_count: Optional total available variant count for the turn.
            selected_variant_id: Optional selected variant ID.
            thinking_blocks_json: Optional canonical thinking evidence.
            assistant_generation_state: Portable assistant generation lifecycle state.
            base_version: Optional previous payload hash for versioned updates.
            entity_version: Optional explicit entity version after the mutation.

        Returns:
            A status mapping. Enqueued results include the durable outbox entry;
            skipped results include a reason describing the missing prerequisite.
        """
        if thinking_blocks_json is not None:
            raise ValueError(
                "Thinking requires committed-intent reconciliation before Sync."
            )

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
            thinking_blocks_json=thinking_blocks_json,
            assistant_generation_state=assistant_generation_state,
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
