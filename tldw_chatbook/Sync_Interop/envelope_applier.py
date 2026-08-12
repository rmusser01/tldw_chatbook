"""Apply pulled Sync v2 envelopes through local domain adapters."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    read_provider_continuation_json,
)
from tldw_chatbook.Sync_Interop.domain_adapters import (
    ChatSyncAdapter,
    MediaSyncAdapter,
    NotesSyncAdapter,
    SourceCacheSyncAdapter,
    WorkspacesSyncAdapter,
)
from tldw_chatbook.Sync_Interop.domain_adapters.notes_m1 import NotesM1SyncAdapter

if TYPE_CHECKING:
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
        if self.dataset_key is None:
            return self._record_conflict(
                envelope,
                conflict_type="missing_dataset_key",
                message="dataset_key is required to apply encrypted Sync v2 envelopes.",
            )
        local_store = self.local_store
        chat_store = None
        if envelope.domain == "chat":
            chat_store = _ContinuationValidatingChatStore(local_store)
            local_store = chat_store
        result = adapter.apply(
            envelope,
            dataset_key=self.dataset_key,
            local_store=local_store,
            record_conflict=self._record_conflict,
        )
        if chat_store is not None and chat_store.warning is not None:
            return {**result, "warning": chat_store.warning}
        return result

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


class _ContinuationValidatingChatStore:
    """Validate private Chat payload data before forwarding one whole record."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.warning: str | None = None

    def get_chat_message_hash(self, stable_key: str) -> str | None:
        reader = getattr(self.store, "get_chat_message_hash", None)
        return reader(stable_key) if callable(reader) else None

    def append_chat_message(
        self,
        stable_key: str,
        payload: dict[str, Any],
        payload_hash: str,
    ) -> None:
        private_value = payload.get("provider_continuation_json")
        if private_value:
            safe_read = read_provider_continuation_json(private_value)
            if payload.get("role") == "assistant" and safe_read.checkpoint is not None:
                payload = {
                    **payload,
                    "provider_continuation_json": dump_provider_continuation_json(
                        safe_read.checkpoint
                    ),
                }
            else:
                payload = dict(payload)
                payload.pop("provider_continuation_json", None)
                self.warning = safe_read.warning or (
                    "Exact tool continuation was discarded."
                )
        elif "provider_continuation_json" in payload:
            payload = dict(payload)
            payload.pop("provider_continuation_json", None)

        writer = getattr(self.store, "append_chat_message", None)
        if callable(writer):
            writer(stable_key, payload, payload_hash)
