"""Apply pulled Sync v2 envelopes through local domain adapters."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Sync_Interop.domain_adapters import (
    ChatSyncAdapter,
    MediaSyncAdapter,
    NotesSyncAdapter,
    SourceCacheSyncAdapter,
    WorkspacesSyncAdapter,
)
from tldw_chatbook.Sync_Interop.domain_adapters.notes_m1 import NotesM1SyncAdapter
from tldw_chatbook.tldw_api import SyncV2Envelope


class SyncEnvelopeApplier:
    """Route Sync v2 envelopes to small local domain adapters."""

    def __init__(
        self,
        *,
        local_store: Any,
        dataset_key: bytes | None = None,
        notes_mirror: Any = None,
        dataset_id: str | None = None,
    ) -> None:
        self.dataset_key = dataset_key
        self.local_store = local_store
        self.notes_mirror = notes_mirror
        self.dataset_id = dataset_id
        self.conflicts: list[dict[str, Any]] = []
        self._adapters: dict[str, Any] = {
            "notes": NotesSyncAdapter(),
            "chat": ChatSyncAdapter(),
            "workspaces": WorkspacesSyncAdapter(),
            "source_cache": SourceCacheSyncAdapter(),
            "media": MediaSyncAdapter(),
            "notes.note": NotesM1SyncAdapter(),
        }

    def apply(self, envelope: SyncV2Envelope) -> dict[str, Any]:
        adapter = self._adapters.get(envelope.domain)
        if adapter is None:
            return self._record_conflict(
                envelope,
                conflict_type="unsupported_domain",
                message=f"Unsupported Sync v2 domain: {envelope.domain}",
            )
        if isinstance(adapter, NotesM1SyncAdapter):
            return adapter.apply(
                envelope,
                local_store=self.local_store,
                notes_mirror=self.notes_mirror,
                dataset_id=self.dataset_id,
                record_conflict=self._record_conflict,
            )
        return adapter.apply(
            envelope,
            dataset_key=self.dataset_key,
            local_store=self.local_store,
            record_conflict=self._record_conflict,
        )

    def _record_conflict(
        self,
        envelope: SyncV2Envelope,
        *,
        conflict_type: str,
        message: str | None = None,
    ) -> dict[str, Any]:
        conflict = {
            "domain": envelope.domain,
            "entity_id": envelope.entity_id,
            "stable_key": envelope.stable_key,
            "client_envelope_id": envelope.client_envelope_id,
            "conflict_type": conflict_type,
        }
        if message:
            conflict["message"] = message
        self.conflicts.append(conflict)
        record = getattr(self.local_store, "record_conflict", None)
        if callable(record):
            record(conflict)
        return {"status": "conflict", "conflict": conflict}
