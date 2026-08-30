"""Idempotently project encrypted profile outbox entries into Sync state."""

from __future__ import annotations

import json
from typing import Any

from tldw_chatbook.Personal_Context.repository import ProfileIntegrityError
from tldw_chatbook.tldw_api import SyncV2Envelope

from .personal_context_adapter import PersonalContextSyncValidationError


class PersonalContextOutboxDispatcher:
    """Cross the profile/Sync database boundary with replay-safe identities."""

    def __init__(self, *, profile_outbox: Any, state_repository: Any, adapter: Any) -> None:
        self.profile_outbox = profile_outbox
        self.state_repository = state_repository
        self.adapter = adapter

    def dispatch_pending(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        device_id: str,
        storage_key: bytes,
        limit: int = 100,
    ) -> dict[str, int]:
        """Copy pending entries, then receipt and shred each source body."""

        dispatched = 0
        quarantined = 0
        for entry in self.profile_outbox.list_pending(limit=limit):
            client_envelope_id = f"personal-context:{entry.outbox_id}"
            try:
                body = self.profile_outbox.read_body(entry.outbox_id)
            except (
                json.JSONDecodeError,
                ProfileIntegrityError,
                TypeError,
                UnicodeDecodeError,
            ):
                body = None
            if body is None:
                self.profile_outbox.quarantine(
                    entry.outbox_id, "encrypted_body_unavailable"
                )
                quarantined += 1
                continue
            try:
                envelope = self.adapter.build_envelope(
                    entry=entry,
                    body=body,
                    dataset_id=dataset_id,
                    device_id=device_id,
                )
            except PersonalContextSyncValidationError:
                self.profile_outbox.quarantine(
                    entry.outbox_id, "invalid_canonical_object"
                )
                quarantined += 1
                continue
            existing = self.state_repository.get_sync_v2_outbox_entry(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
                client_envelope_id=client_envelope_id,
            )
            if existing is not None:
                sync_head = self.state_repository.get_sync_v2_remote_head(
                    server_profile_id=server_profile_id,
                    authenticated_principal_id=authenticated_principal_id,
                    workspace_scope=workspace_scope,
                    dataset_id=dataset_id,
                    domain=envelope.domain,
                    object_id=str(envelope.object_id),
                )
                expected = self.adapter.build_envelope(
                    entry=entry,
                    body=body,
                    dataset_id=dataset_id,
                    device_id=device_id,
                    sync_head=sync_head,
                )
                try:
                    staged = SyncV2Envelope.model_validate(existing["envelope"])
                    restored = self.adapter.restore_from_storage(
                        staged,
                        storage_key=storage_key,
                    )
                    if not _same_staged_copy(restored, expected):
                        raise PersonalContextSyncValidationError(
                            "personal_context_storage_invalid"
                        )
                except (PersonalContextSyncValidationError, TypeError, ValueError):
                    self.profile_outbox.quarantine(
                        entry.outbox_id,
                        "destination_copy_invalid",
                        preserve_body=True,
                    )
                    quarantined += 1
                    continue
                self.profile_outbox.acknowledge(entry.outbox_id, client_envelope_id)
                dispatched += 1
                continue
            if self.state_repository.has_pending_sync_v2_object(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
                domain=envelope.domain,
                object_id=str(envelope.object_id),
                exclude_client_envelope_id=envelope.client_envelope_id,
            ):
                continue
            sync_head = self.state_repository.get_sync_v2_remote_head(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
                domain=envelope.domain,
                object_id=str(envelope.object_id),
            )
            envelope = self.adapter.build_envelope(
                entry=entry,
                body=body,
                dataset_id=dataset_id,
                device_id=device_id,
                sync_head=sync_head,
            )
            stored_envelope = self.adapter.protect_for_storage(
                envelope,
                storage_key=storage_key,
            )
            self.state_repository.enqueue_sync_v2_outbox_envelope(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
                envelope=stored_envelope,
            )
            self._after_destination_enqueue(entry, stored_envelope)
            self.profile_outbox.acknowledge(
                entry.outbox_id, envelope.client_envelope_id
            )
            dispatched += 1
        return {"dispatched": dispatched, "quarantined": quarantined}

    @staticmethod
    def _after_destination_enqueue(_entry: Any, _envelope: Any) -> None:
        """Test seam for a crash after the durable destination commit."""


def _same_staged_copy(
    restored: SyncV2Envelope,
    expected: SyncV2Envelope,
) -> bool:
    """Compare the complete normalized client envelope before source shredding."""

    return restored.model_dump(mode="json") == expected.model_dump(mode="json")


__all__ = ["PersonalContextOutboxDispatcher"]
