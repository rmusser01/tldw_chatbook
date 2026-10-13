"""Focused notes.note push/pull/apply flow against an M1 Sync v2 server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope


class NotesM1SyncFlow:

    def __init__(
        self,
        *,
        client,
        builder,
        mirror,
        local_store,
        dataset_id: str,
        device_id: str,
        assert_authority_current: Callable[[], None] | None = None,
    ) -> None:
        self.assert_authority_current = assert_authority_current
        self.client = client
        self.builder = builder
        self.mirror = mirror
        self.local_store = local_store
        self.dataset_id = dataset_id
        self.device_id = device_id
        self._client_sequence = 0

    async def push(self, envelopes: list[SyncV2Envelope]) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from tldw_chatbook.tldw_api import SyncV2PushRequest

        client = self.client
        if (
            any(env.adapter_version == 2 for env in envelopes)
            and self.assert_authority_current is None
        ):
            raise ValueError("notes_authority_unavailable")
        self._assert_current(client)
        capabilities = await client.get_sync_v2_capabilities(dataset_id=self.dataset_id)
        self._assert_current(client)
        capability_record = (
            capabilities.model_dump()
            if hasattr(capabilities, "model_dump")
            else capabilities
        )
        writable = (
            capability_record.get("writable_adapter_versions", {})
            if isinstance(capability_record, dict)
            else {}
        )
        if any(
            env.domain != "notes.note"
            or env.adapter_version not in writable.get("notes.note", [])
            for env in envelopes
        ):
            raise ValueError("notes_language_upgrade_required")
        self.builder.notes_adapter_versions = writable
        for env in envelopes:
            self._client_sequence += 1
            env.client_sequence = self._client_sequence
        request = SyncV2PushRequest(
            dataset_id=self.dataset_id, device_id=self.device_id, envelopes=envelopes
        )
        self._assert_current(client)
        response = await client.push_sync_v2_envelopes(request)
        self._assert_current(client)
        payload_hashes = {env.client_envelope_id: env.payload_hash for env in envelopes}
        submitted = {env.client_envelope_id: env for env in envelopes}
        mirror_errors = 0
        for acknowledgement in [
            *response.accepted,
            *(getattr(response, "idempotent", []) or []),
        ]:
            env = submitted.get(getattr(acknowledgement, "client_envelope_id", ""))
            if (
                env is not None
                and env.adapter_version == 2
                and not self._complete_acknowledgement(env, acknowledgement)
            ):
                mirror_errors += 1
                continue
            if not self._record_acknowledgement(acknowledgement, payload_hashes):
                mirror_errors += 1
        return {
            "accepted": len(response.accepted),
            "idempotent": len(getattr(response, "idempotent", []) or []),
            "rejected": len(getattr(response, "rejected", []) or []),
            "conflicts": len(getattr(response, "conflicts", []) or []),
            "apply_errors": len(getattr(response, "apply_errors", []) or []),
            "mirror_errors": mirror_errors,
            "next_cursor": response.next_cursor,
        }

    async def pull(self, *, cursor: int) -> dict[str, Any]:
        client = self.client
        self._assert_current(client)
        response = await client.pull_sync_v2_envelopes(
            dataset_id=self.dataset_id,
            device_id=self.device_id,
            cursor=str(cursor),
            domains=["notes.note"],
        )
        self._assert_current(client)
        applier = SyncEnvelopeApplier(
            local_store=self.local_store,
            notes_mirror=self.mirror,
            dataset_id=self.dataset_id,
        )
        applied = noop = conflicts = 0
        for env in response.envelopes:
            status = applier.apply(env).get("status")
            if status == "applied":
                applied += 1
            elif status == "noop":
                noop += 1
            else:
                conflicts += 1
        return {
            "applied": applied,
            "noop": noop,
            "conflicts": conflicts,
            "next_cursor": response.next_cursor,
            "has_more": response.has_more,
        }

    def _assert_current(self, client: Any) -> None:
        if self.client is not client:
            raise ValueError("notes_authority_changed")
        if self.assert_authority_current is not None:
            self.assert_authority_current()

    @staticmethod
    def _complete_acknowledgement(
        envelope: SyncV2Envelope, acknowledgement: Any
    ) -> bool:
        revision = getattr(acknowledgement, "object_revision", None)
        cursor = getattr(acknowledgement, "server_cursor", None)
        if cursor is None:
            cursor = getattr(acknowledgement, "server_sequence", None)
        return (
            getattr(acknowledgement, "domain", None) == "notes.note"
            and getattr(acknowledgement, "object_id", None) == envelope.object_id
            and getattr(acknowledgement, "apply_status", None) == "applied"
            and type(revision) is int
            and revision == (envelope.base_object_revision or 0) + 1
            and type(cursor) is int
            and cursor > 0
        )

    def _record_acknowledgement(
        self, acknowledgement: Any, payload_hashes: dict[str, str]
    ) -> bool:
        object_id = getattr(acknowledgement, "object_id", None) or getattr(
            acknowledgement, "entity_id", None
        )
        if object_id is None:
            return False

        existing = self.mirror.get(self.dataset_id, object_id)
        object_revision = getattr(acknowledgement, "object_revision", None)
        server_cursor = getattr(acknowledgement, "server_cursor", None)
        if server_cursor is None:
            server_cursor = getattr(acknowledgement, "server_sequence", None)

        payload_hash = payload_hashes.get(
            getattr(acknowledgement, "client_envelope_id", "")
        )
        if existing is not None and (object_revision is None or server_cursor is None):
            object_revision = (
                existing.object_revision
                if object_revision is None
                else max(existing.object_revision, object_revision)
            )
            server_cursor = (
                existing.server_cursor
                if server_cursor is None
                else max(existing.server_cursor, server_cursor)
            )
            payload_hash = existing.object_hash
        elif payload_hash is None or object_revision is None or server_cursor is None:
            return False
        elif existing is not None:
            object_revision = max(existing.object_revision, object_revision)
            server_cursor = max(existing.server_cursor, server_cursor)

        self.mirror.record(
            self.dataset_id,
            object_id,
            object_revision=object_revision,
            object_hash=payload_hash,
            server_cursor=server_cursor,
        )
        return True
